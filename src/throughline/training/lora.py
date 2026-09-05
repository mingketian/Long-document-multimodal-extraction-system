"""LoRA fine-tuning for Qwen2.5-VL.

Full fine-tuning of a 7B vision-language model to specialise it on one extraction
schema is the wrong trade: it costs an order of magnitude more compute, produces a
checkpoint per document type, and risks degrading the general document understanding
that made the base model worth starting from. LoRA gives task specialisation in
adapter weights measured in tens of megabytes, and the base checkpoint stays shared
across every schema.

Three choices here are deliberate and worth defending:

**The vision tower is frozen.** The adapter targets the language model's attention
and MLP projections only. The task is not "learn to see documents" - the base model
already does - it is "learn to emit this schema, from these page groups, with these
citations". Adapting the vision encoder spends parameters on a problem that is not
the bottleneck.

**Loss is computed on the completion only.** The prompt contains the schema and the
whole page layout; training the model to reproduce *that* would waste most of the
gradient on text it will always be given. :class:`CompletionOnlyCollator` masks
everything up to the assistant turn.

**Sequence length is generous.** A page-group prompt with layout text for four pages
is long. Truncating it silently drops the end of the last page, which is exactly
where a continued table lives.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class LoraConfig:
    """LoRA hyperparameters.

    Args:
        r: Adapter rank. 16 is the working default; 8 underfits table continuation,
            32 showed no gain worth the extra memory.
        alpha: Scaling factor, conventionally ``2 * r``.
        dropout: Adapter dropout.
        target_modules: Projections to adapt. Attention plus MLP; no vision modules.
        modules_to_save: Extra modules trained in full, if any.
        bias: Whether to train bias terms.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    modules_to_save: tuple[str, ...] = ()
    bias: str = "none"

    def to_peft(self) -> Any:
        """Build the PEFT ``LoraConfig`` this describes."""
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            modules_to_save=list(self.modules_to_save) or None,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
        }


@dataclass
class TrainingConfig:
    """Everything about a fine-tuning run."""

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir: str = "outputs/lora-run"
    lora: LoraConfig = field(default_factory=LoraConfig)

    epochs: float = 2.0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"

    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    """Effective batch 16 on one GPU. Page-group prompts are long; this is how the
    effective batch is reached without exceeding memory."""

    max_seq_length: int = 8_192
    gradient_checkpointing: bool = True
    bf16: bool = True

    freeze_vision_tower: bool = True
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    seed: int = 13

    experiment: str = "throughline"
    run_name: str = "lora"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.per_device_batch_size
            * self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "freeze_vision_tower": self.freeze_vision_tower,
            "bf16": self.bf16,
            "seed": self.seed,
            **{f"lora_{k}": v for k, v in self.lora.to_dict().items()},
        }


class CompletionOnlyCollator:
    """Mask the prompt so loss is computed on the assistant turn only.

    Finds the assistant-turn marker in each tokenised example and sets every label
    before it to ``-100``. Without this the model spends most of its gradient
    learning to reproduce the schema block and the page layout it is handed at
    inference time anyway.
    """

    def __init__(self, processor: Any, response_marker: str = "<|im_start|>assistant") -> None:
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.response_ids = self.tokenizer(
            response_marker, add_special_tokens=False
        )["input_ids"]

    def _mask(self, input_ids: Sequence[int], labels: list[int]) -> list[int]:
        marker = self.response_ids
        span = len(marker)
        cut = None
        for index in range(len(input_ids) - span, -1, -1):
            if list(input_ids[index : index + span]) == marker:
                cut = index + span
                break
        if cut is None:
            LOGGER.debug("Assistant marker not found; training on the full sequence.")
            return labels
        return [-100] * cut + labels[cut:]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        masked = [
            self._mask(batch["input_ids"][row].tolist(), labels[row].tolist())
            for row in range(labels.size(0))
        ]
        batch["labels"] = torch.tensor(masked, dtype=labels.dtype)
        return batch


def _require(package: str, extra: str) -> Any:
    try:
        return __import__(package)
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError(
            f"Fine-tuning needs {package}. Install with: pip install 'throughline[{extra}]'"
        ) from exc


def load_model_for_training(config: TrainingConfig) -> tuple[Any, Any]:
    """Load the base model with the LoRA adapter attached and the vision tower frozen."""
    _require("torch", "train")
    _require("transformers", "train")
    _require("peft", "train")

    import torch
    from peft import get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.freeze_vision_tower:
        frozen = 0
        for name, parameter in model.named_parameters():
            if "visual" in name or "vision_tower" in name:
                parameter.requires_grad = False
                frozen += parameter.numel()
        LOGGER.info("Froze %.1fM vision parameters", frozen / 1e6)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, config.lora.to_peft())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        "Trainable: %.2fM / %.1fM (%.3f%%)",
        trainable / 1e6,
        total / 1e6,
        100 * trainable / total,
    )

    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    return model, processor


def train(
    config: TrainingConfig,
    train_path: str | Path,
    validation_path: str | Path | None = None,
) -> str:
    """Run supervised fine-tuning and return the adapter directory.

    Args:
        config: Training configuration.
        train_path: JSONL written by :func:`throughline.training.dataset.write_jsonl`.
        validation_path: Optional held-out JSONL, split by document.

    Returns:
        Path to the saved LoRA adapter.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]
    from transformers import Trainer, TrainingArguments

    from throughline.evaluation import mlflow_tracking

    model, processor = load_model_for_training(config)

    data_files = {"train": str(train_path)}
    if validation_path:
        data_files["validation"] = str(validation_path)
    raw = load_dataset("json", data_files=data_files)

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = processor.apply_chat_template(example["messages"], tokenize=False)
        encoded = processor.tokenizer(
            text, truncation=True, max_length=config.max_seq_length
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    tokenized = raw.map(tokenize, remove_columns=raw["train"].column_names)

    arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if validation_path else "no",
        eval_steps=config.eval_steps if validation_path else None,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=CompletionOnlyCollator(processor),
    )

    with mlflow_tracking.run_context(config.run_name, experiment=config.experiment):
        mlflow_tracking.log_params(config.to_dict())
        result = trainer.train()
        mlflow_tracking.log_metrics(
            {
                "train_loss": float(result.training_loss),
                "train_runtime_seconds": float(result.metrics.get("train_runtime", 0.0)),
            }
        )

        adapter_dir = Path(config.output_dir) / "adapter"
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        (adapter_dir / "throughline_training_config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        mlflow_tracking.log_artifact(str(adapter_dir))

    LOGGER.info("Adapter saved to %s", adapter_dir)
    return str(adapter_dir)
