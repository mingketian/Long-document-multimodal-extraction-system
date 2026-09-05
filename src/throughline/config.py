"""Run configuration.

One file describes a whole run: which schema, which backend, how pages are grouped,
when to stop, and where the caches live. Keeping it in one declarative object rather
than scattered across call sites is what makes two runs comparable - the config is
logged verbatim to MLflow, so "the balanced profile on the invoice schema" is a
reproducible statement rather than a description of what someone remembers doing.

Three named profiles cover the operating points that matter:

* :data:`ACCURACY` - read everything, cache nothing, measure the ceiling.
* :data:`BALANCED` - the production default; early exit on a satisfied, evidenced schema.
* :data:`FAST` - triage: aggressive exit, smaller windows, evidence not required.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from throughline.grouping.page_groups import GroupingConfig
from throughline.models.base import GenerationConfig
from throughline.pipeline.early_exit import EarlyExitPolicy


@dataclass
class BackendConfig:
    """Which model serves the run."""

    kind: str = "rule-based"
    """One of ``rule-based``, ``qwen-vl``, ``sagemaker``, ``sagemaker-async``."""

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path: str | None = None
    endpoint_name: str | None = None
    region: str = "us-east-1"
    bucket: str | None = None
    device_map: str = "auto"
    dtype: str = "bfloat16"

    def build(self) -> Any:
        """Instantiate the backend this describes."""
        if self.kind == "rule-based":
            from throughline.models.rule_based import RuleBasedBackend

            return RuleBasedBackend()

        if self.kind == "qwen-vl":
            from throughline.models.qwen_vl import QwenVLBackend

            return QwenVLBackend(
                model_id=self.model_id,
                adapter_path=self.adapter_path,
                device_map=self.device_map,
                dtype=self.dtype,
            )

        if self.kind == "sagemaker":
            from throughline.models.sagemaker import SageMakerBackend

            if not self.endpoint_name:
                raise ValueError("backend.kind='sagemaker' requires endpoint_name.")
            return SageMakerBackend(endpoint_name=self.endpoint_name, region=self.region)

        if self.kind == "sagemaker-async":
            from throughline.models.sagemaker import SageMakerAsyncBackend

            if not self.endpoint_name or not self.bucket:
                raise ValueError(
                    "backend.kind='sagemaker-async' requires endpoint_name and bucket."
                )
            return SageMakerAsyncBackend(
                endpoint_name=self.endpoint_name, bucket=self.bucket, region=self.region
            )

        raise ValueError(f"Unknown backend kind: {self.kind!r}")


@dataclass
class CacheConfig:
    """Where the two content-addressed caches live."""

    enabled: bool = True
    ocr_dir: str = ".cache/ocr"
    prompt_dir: str = ".cache/prompts"
    ttl_seconds: float | None = None

    def build_prompt_cache(self) -> Any:
        from throughline.caching.store import PromptCache

        return PromptCache(
            cache_dir=self.prompt_dir, enabled=self.enabled, ttl_seconds=self.ttl_seconds
        )


@dataclass
class RunConfig:
    """A complete, serialisable description of one extraction run."""

    schema: str = "invoice"
    backend: BackendConfig = field(default_factory=BackendConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    early_exit: EarlyExitPolicy = field(default_factory=EarlyExitPolicy)
    cache: CacheConfig = field(default_factory=CacheConfig)
    use_retrieval: bool = True
    max_chars_per_page: int = 6_000
    repair_attempts: int = 1
    profile: str = "custom"

    # ── construction ──────────────────────────────────────────────────
    def build_pipeline(self) -> Any:
        """Build the :class:`~throughline.pipeline.orchestrator.ExtractionPipeline`."""
        from throughline.pipeline.orchestrator import ExtractionPipeline, PipelineConfig
        from throughline.schema import registry

        return ExtractionPipeline(
            backend=self.backend.build(),
            schema=registry.get(self.schema),
            config=PipelineConfig(
                grouping=self.grouping,
                generation=self.generation,
                early_exit=self.early_exit,
                use_retrieval=self.use_retrieval,
                max_chars_per_page=self.max_chars_per_page,
                repair_attempts=self.repair_attempts,
                prompt_cache=self.cache.build_prompt_cache() if self.cache.enabled else None,
            ),
        )

    # ── serialisation ─────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # EarlyExitPolicy carries private run state; drop it from the record.
        payload["early_exit"] = {
            key: value
            for key, value in payload["early_exit"].items()
            if not key.startswith("_")
        }
        return payload

    def flat_params(self) -> dict[str, Any]:
        """Flattened key/value pairs, for MLflow params."""
        flat: dict[str, Any] = {"schema": self.schema, "profile": self.profile}
        for section, values in self.to_dict().items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat[f"{section}.{key}"] = value
            elif section not in flat:
                flat[section] = values
        return flat

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        def build(target: type, raw: Any) -> Any:
            if not isinstance(raw, dict):
                return target()
            allowed = {f.name for f in target.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            return target(**{k: v for k, v in raw.items() if k in allowed})

        return cls(
            schema=payload.get("schema", "invoice"),
            backend=build(BackendConfig, payload.get("backend")),
            grouping=build(GroupingConfig, payload.get("grouping")),
            generation=build(GenerationConfig, payload.get("generation")),
            early_exit=build(EarlyExitPolicy, payload.get("early_exit")),
            cache=build(CacheConfig, payload.get("cache")),
            use_retrieval=bool(payload.get("use_retrieval", True)),
            max_chars_per_page=int(payload.get("max_chars_per_page", 6_000)),
            repair_attempts=int(payload.get("repair_attempts", 1)),
            profile=payload.get("profile", "custom"),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> RunConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_env(cls, base: RunConfig | None = None) -> RunConfig:
        """Overlay ``THROUGHLINE_*`` environment variables onto a config.

        Recognised: ``THROUGHLINE_SCHEMA``, ``THROUGHLINE_BACKEND``,
        ``THROUGHLINE_ENDPOINT``, ``THROUGHLINE_ADAPTER``, ``THROUGHLINE_REGION``,
        ``THROUGHLINE_CACHE`` (``0`` disables).
        """
        config = base or cls()
        backend = replace(
            config.backend,
            kind=os.environ.get("THROUGHLINE_BACKEND", config.backend.kind),
            endpoint_name=os.environ.get("THROUGHLINE_ENDPOINT", config.backend.endpoint_name),
            adapter_path=os.environ.get("THROUGHLINE_ADAPTER", config.backend.adapter_path),
            region=os.environ.get("THROUGHLINE_REGION", config.backend.region),
        )
        cache = replace(
            config.cache,
            enabled=os.environ.get("THROUGHLINE_CACHE", "1") not in {"0", "false", "False"},
        )
        return replace(
            config,
            schema=os.environ.get("THROUGHLINE_SCHEMA", config.schema),
            backend=backend,
            cache=cache,
        )


# ── named profiles ────────────────────────────────────────────────────
ACCURACY = RunConfig(
    profile="accuracy",
    grouping=GroupingConfig(max_pages=4, overlap=1),
    early_exit=EarlyExitPolicy(enabled=False),
    use_retrieval=False,
    repair_attempts=2,
)
"""Read every group in page order. The accuracy ceiling a speedup is measured against."""

BALANCED = RunConfig(
    profile="balanced",
    grouping=GroupingConfig(max_pages=4, overlap=1),
    early_exit=EarlyExitPolicy(
        enabled=True, min_groups=2, require_valid=True, require_evidence=True, patience=2
    ),
    use_retrieval=True,
    repair_attempts=1,
)
"""Production default: stop once the schema is satisfied, valid and evidenced."""

FAST = RunConfig(
    profile="fast",
    grouping=GroupingConfig(max_pages=3, overlap=1),
    early_exit=EarlyExitPolicy(
        enabled=True, min_groups=1, require_valid=True, require_evidence=False, patience=1
    ),
    use_retrieval=True,
    repair_attempts=0,
)
"""Triage: lowest latency, no evidence requirement. Not for a system of record."""

PROFILES: dict[str, RunConfig] = {
    "accuracy": ACCURACY,
    "balanced": BALANCED,
    "fast": FAST,
}


def profile(name: str) -> RunConfig:
    """Look up a named profile."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"Unknown profile {name!r}. Available: {sorted(PROFILES)}") from None
