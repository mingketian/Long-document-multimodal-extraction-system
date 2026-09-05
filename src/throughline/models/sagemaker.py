"""SageMaker backends.

Two shapes, because the workload has two shapes.

:class:`SageMakerBackend` calls a real-time endpoint. That is what the interactive
extraction path uses: one page group per request, latency in the seconds, autoscaled
behind the endpoint.

:class:`SageMakerAsyncBackend` calls an asynchronous endpoint via S3. Long documents
processed in bulk do not need sub-second turnaround, and the async endpoint accepts
much larger payloads and scales to zero between batches - which for a corpus that
arrives in weekly drops rather than continuously is the difference between paying
for a warm GPU all week and paying for the hours it actually runs.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
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


def _encode_image(path: str) -> str | None:
    file = Path(path)
    if not file.exists():
        LOGGER.warning("Page image missing, skipping: %s", path)
        return None
    return base64.b64encode(file.read_bytes()).decode("ascii")


def _build_payload(prompt: PromptBundle, config: GenerationConfig) -> dict[str, Any]:
    """The request body the inference handler expects.

    Kept in OpenAI-ish chat shape so the same handler serves both a HuggingFace TGI
    container and a custom Qwen2.5-VL container without a translation layer.
    """
    images = [encoded for path in prompt.image_paths if (encoded := _encode_image(path))]
    return {
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "images": images,
        "parameters": config.to_dict(),
    }


def _extract_text(payload: Any) -> str:
    """Pull generated text out of the several shapes SageMaker containers return."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list) and payload:
        return _extract_text(payload[0])
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "output", "completion"):
            if key in payload:
                return _extract_text(payload[key])
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
                if "text" in choice:
                    return str(choice["text"])
    raise BackendError(f"Could not find generated text in response: {str(payload)[:300]}")


@dataclass
class SageMakerBackend:
    """Invoke a real-time SageMaker endpoint."""

    endpoint_name: str
    region: str = "us-east-1"
    content_type: str = "application/json"
    max_retries: int = 3
    retry_base_delay: float = 1.0
    name: str = "sagemaker-realtime"

    _client: Any = field(default=None, init=False, repr=False)

    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency-gated
                raise BackendError(
                    "SageMakerBackend needs boto3. Install with: pip install 'throughline[aws]'"
                ) from exc
            self._client = boto3.client("sagemaker-runtime", region_name=self.region)
        return self._client

    @timed
    def generate(
        self, prompt: PromptBundle, config: GenerationConfig | None = None
    ) -> GenerationResult:
        config = config or GenerationConfig()
        body = json.dumps(_build_payload(prompt, config)).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client().invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType=self.content_type,
                    Accept="application/json",
                    Body=body,
                )
                payload = json.loads(response["Body"].read().decode("utf-8"))
                text = _extract_text(payload)
                return GenerationResult(
                    text=text.strip(),
                    prompt_tokens=int(payload.get("prompt_tokens", 0))
                    if isinstance(payload, dict)
                    else 0,
                    completion_tokens=int(payload.get("completion_tokens", 0))
                    if isinstance(payload, dict)
                    else 0,
                    backend=self.name,
                    metadata={"endpoint": self.endpoint_name},
                )
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised after
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                delay = self.retry_base_delay * (2**attempt)
                LOGGER.warning(
                    "Endpoint %s attempt %s/%s failed (%s); retrying in %.1fs",
                    self.endpoint_name,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise BackendError(
            f"Endpoint {self.endpoint_name} failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


@dataclass
class SageMakerAsyncBackend:
    """Invoke an asynchronous SageMaker endpoint through S3.

    The request is written to ``s3://{bucket}/{input_prefix}/`` and the endpoint
    writes its response to the output location it returns. Polling is bounded by
    ``timeout_seconds`` so a stuck job fails the document rather than the batch.
    """

    endpoint_name: str
    bucket: str
    input_prefix: str = "throughline/async-input"
    region: str = "us-east-1"
    poll_seconds: float = 5.0
    timeout_seconds: float = 900.0
    name: str = "sagemaker-async"

    _runtime: Any = field(default=None, init=False, repr=False)
    _s3: Any = field(default=None, init=False, repr=False)

    def _clients(self) -> tuple[Any, Any]:
        if self._runtime is None or self._s3 is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency-gated
                raise BackendError(
                    "SageMakerAsyncBackend needs boto3. "
                    "Install with: pip install 'throughline[aws]'"
                ) from exc
            self._runtime = boto3.client("sagemaker-runtime", region_name=self.region)
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._runtime, self._s3

    @timed
    def generate(
        self, prompt: PromptBundle, config: GenerationConfig | None = None
    ) -> GenerationResult:
        config = config or GenerationConfig()
        runtime, s3 = self._clients()

        key = f"{self.input_prefix}/{uuid.uuid4().hex}.json"
        s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(_build_payload(prompt, config)).encode("utf-8"),
            ContentType="application/json",
        )

        response = runtime.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=f"s3://{self.bucket}/{key}",
            ContentType="application/json",
        )
        output_location = response["OutputLocation"]
        payload = self._await_output(s3, output_location)

        return GenerationResult(
            text=_extract_text(payload).strip(),
            backend=self.name,
            metadata={"endpoint": self.endpoint_name, "output_location": output_location},
        )

    def _await_output(self, s3: Any, location: str) -> Any:
        _, _, remainder = location.partition("s3://")
        bucket, _, key = remainder.partition("/")
        deadline = time.monotonic() + self.timeout_seconds

        while time.monotonic() < deadline:
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                return json.loads(obj["Body"].read().decode("utf-8"))
            except s3.exceptions.NoSuchKey:
                time.sleep(self.poll_seconds)

        raise BackendError(
            f"Async endpoint {self.endpoint_name} produced no output within "
            f"{self.timeout_seconds:.0f}s ({location})."
        )
