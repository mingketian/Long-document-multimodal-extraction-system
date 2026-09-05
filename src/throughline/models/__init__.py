"""VLM backends.

Heavy backends are imported lazily so that ``import throughline`` costs nothing and
works without torch, transformers or boto3 installed.
"""

from typing import Any

from throughline.models.base import (
    BackendError,
    GenerationConfig,
    GenerationResult,
    VLMBackend,
)
from throughline.models.rule_based import EchoBackend, RuleBasedBackend

__all__ = [
    "BackendError",
    "EchoBackend",
    "GenerationConfig",
    "GenerationResult",
    "QwenVLBackend",
    "RuleBasedBackend",
    "SageMakerAsyncBackend",
    "SageMakerBackend",
    "VLMBackend",
]

_LAZY = {
    "QwenVLBackend": ("throughline.models.qwen_vl", "QwenVLBackend"),
    "SageMakerBackend": ("throughline.models.sagemaker", "SageMakerBackend"),
    "SageMakerAsyncBackend": ("throughline.models.sagemaker", "SageMakerAsyncBackend"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attribute = _LAZY[name]
        return getattr(importlib.import_module(module_name), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
