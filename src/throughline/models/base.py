"""The VLM backend interface.

Everything above this line in the stack - grouping, state, prompting, attribution -
is backend-agnostic. A backend takes a :class:`PromptBundle` and returns text. That
is the whole contract, and keeping it that narrow is what let the same pipeline run
against a local Qwen2.5-VL checkpoint during development, a SageMaker endpoint in
the sandbox, and a deterministic stub in CI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from throughline.prompting.templates import PromptBundle


@dataclass
class GenerationConfig:
    """Decoding parameters.

    Defaults are the extraction-appropriate ones: greedy, because extraction has a
    single correct answer and sampling only adds variance; and a no-repeat n-gram
    guard, because long-horizon multimodal decoding over a table is exactly where
    autoregressive models fall into repetition loops.
    """

    max_new_tokens: int = 4_096
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    no_repeat_ngram_size: int = 0
    """0 disables. 20-40 is a reasonable band for long tabular output."""

    repetition_window: int = 0
    """Window the n-gram guard looks back over. 0 means the whole sequence."""

    stop: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_window": self.repetition_window,
        }


@dataclass
class GenerationResult:
    """One backend call: what came back and what it cost."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    cached: bool = False
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class VLMBackend(Protocol):
    """Anything that can turn a prompt bundle into text."""

    name: str

    def generate(
        self, prompt: PromptBundle, config: GenerationConfig | None = None
    ) -> GenerationResult:
        ...


class BackendError(RuntimeError):
    """Raised when a backend fails in a way the pipeline should surface, not retry."""


def timed(function):
    """Decorator recording wall-clock latency onto the returned result."""

    def wrapper(*args: Any, **kwargs: Any) -> GenerationResult:
        started = time.perf_counter()
        result = function(*args, **kwargs)
        if isinstance(result, GenerationResult) and not result.latency_seconds:
            result.latency_seconds = time.perf_counter() - started
        return result

    wrapper.__name__ = getattr(function, "__name__", "wrapper")
    wrapper.__doc__ = function.__doc__
    return wrapper
