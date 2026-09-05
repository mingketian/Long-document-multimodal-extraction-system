"""Qwen2.5-VL backend (local transformers).

Used for development, LoRA evaluation, and any run where the checkpoint is on the
same machine as the pipeline. Production traffic goes through
:class:`~throughline.models.sagemaker.SageMakerBackend` against the same weights.

The class is import-safe without ``torch``/``transformers`` installed: heavy imports
happen inside :meth:`load`, so the module can be imported in CI and by the CLI
without pulling several gigabytes of dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from throughline.models.base import (
    BackendError,
    GenerationConfig,
    GenerationResult,
    timed,
)
from throughline.prompting.templates import PromptBundle

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass
class QwenVLBackend:
    """Run Qwen2.5-VL locally through 🤗 transformers.

    Args:
        model_id: Hub id or local path of the base checkpoint.
        adapter_path: Optional LoRA adapter directory, applied with PEFT. This is how
            a fine-tuned run is evaluated against the same code path as the base.
        device_map: Passed to ``from_pretrained``. ``"auto"`` shards across visible
            GPUs.
        dtype: Torch dtype name. ``bfloat16`` on Ampere and later.
        max_pixels: Upper bound on visual tokens per image. Long documents are the
            reason this matters - 40 pages at full resolution will not fit, and
            capping resolution is a better trade than dropping pages.
        attn_implementation: ``flash_attention_2`` when available.
    """

    model_id: str = DEFAULT_MODEL_ID
    adapter_path: str | None = None
    device_map: str = "auto"
    dtype: str = "bfloat16"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    name: str = "qwen2.5-vl"

    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    # ── lifecycle ─────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Materialise the model and processor. Idempotent."""
        if self.is_loaded:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise BackendError(
                "QwenVLBackend needs torch and transformers. "
                "Install with: pip install 'throughline[vlm]'"
            ) from exc

        LOGGER.info("Loading %s (dtype=%s, device_map=%s)", self.model_id, self.dtype, self.device_map)

        kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, self.dtype),
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)

        if self.adapter_path:
            adapter = Path(self.adapter_path)
            if not adapter.exists():
                raise BackendError(f"LoRA adapter not found: {adapter}")
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - dependency-gated
                raise BackendError(
                    "Loading a LoRA adapter needs peft. "
                    "Install with: pip install 'throughline[train]'"
                ) from exc
            LOGGER.info("Applying LoRA adapter from %s", adapter)
            self._model = PeftModel.from_pretrained(self._model, str(adapter))

        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=self.trust_remote_code,
        )

    def unload(self) -> None:
        """Drop the model and free GPU memory."""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # ── inference ─────────────────────────────────────────────────────────
    def _build_messages(self, prompt: PromptBundle) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": path} for path in prompt.image_paths
        ]
        content.append({"type": "text", "text": prompt.user})
        return [
            {"role": "system", "content": [{"type": "text", "text": prompt.system}]},
            {"role": "user", "content": content},
        ]

    @timed
    def generate(
        self, prompt: PromptBundle, config: GenerationConfig | None = None
    ) -> GenerationResult:
        """Generate one completion for a page group."""
        self.load()
        config = config or GenerationConfig()

        import torch

        messages = self._build_messages(prompt)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        images = self._load_images(prompt.image_paths)
        inputs = self._processor(
            text=[text],
            images=images or None,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generate_kwargs["temperature"] = config.temperature
            generate_kwargs["top_p"] = config.top_p
        if config.no_repeat_ngram_size:
            generate_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size

        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = generated[:, prompt_length:]
        decoded = self._processor.batch_decode(completion_ids, skip_special_tokens=True)[0]

        return GenerationResult(
            text=decoded.strip(),
            prompt_tokens=int(prompt_length),
            completion_tokens=int(completion_ids.shape[1]),
            backend=self.name,
            metadata={"model_id": self.model_id, "adapter": self.adapter_path},
        )

    @staticmethod
    def _load_images(paths: list[str]) -> list[Any]:
        if not paths:
            return []
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise BackendError(
                "Image input needs Pillow. Install with: pip install 'throughline[vlm]'"
            ) from exc
        images = []
        for path in paths:
            if Path(path).exists():
                images.append(Image.open(path).convert("RGB"))
            else:
                LOGGER.warning("Page image missing, continuing text-only: %s", path)
        return images
