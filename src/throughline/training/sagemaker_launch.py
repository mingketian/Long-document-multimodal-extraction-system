"""SageMaker training-job launcher.

Development happens on one GPU; real fine-tuning runs happen on SageMaker so the job
is reproducible, logged, and does not occupy a workstation for hours. This module
builds the job definition and hands it to the SDK.

Nothing here duplicates :mod:`throughline.training.lora` - the entry point that
SageMaker executes *is* that module. The launcher's only job is to describe where the
data lives, what instance to run on, and how to get the adapter back out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from throughline.training.lora import TrainingConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training:2.3.0-transformers4.46.1-gpu-py311-cu121-ubuntu20.04"
)


@dataclass
class SageMakerTrainingJob:
    """A SageMaker training job for the LoRA fine-tune.

    Args:
        role_arn: Execution role with S3 and ECR access.
        train_s3_uri: Prefix holding ``train.jsonl`` (and ``validation.jsonl``).
        output_s3_uri: Where the adapter tarball is written.
        instance_type: ``ml.g5.2xlarge`` fits a 7B LoRA at 8k sequence length;
            ``ml.g5.12xlarge`` shortens wall-clock when the corpus grows.
        use_spot: Spot instances cut cost substantially. Safe here because the
            trainer checkpoints, so an interruption resumes rather than restarts.
    """

    role_arn: str
    train_s3_uri: str
    output_s3_uri: str
    region: str = "us-east-1"
    instance_type: str = "ml.g5.2xlarge"
    instance_count: int = 1
    volume_size_gb: int = 200
    max_runtime_seconds: int = 24 * 3600
    image_uri: str = DEFAULT_IMAGE
    use_spot: bool = True
    max_wait_seconds: int = 36 * 3600
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: dict[str, str] = field(default_factory=dict)

    def hyperparameters(self) -> dict[str, str]:
        """Training config flattened into SageMaker hyperparameters."""
        return {key: str(value) for key, value in self.training.to_dict().items()}

    def estimator(self) -> Any:
        """Build the SageMaker ``Estimator`` for this job."""
        try:
            from sagemaker.estimator import Estimator
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError(
                "Launching needs the sagemaker SDK. "
                "Install with: pip install 'throughline[aws]'"
            ) from exc

        kwargs: dict[str, Any] = {
            "image_uri": self.image_uri,
            "role": self.role_arn,
            "instance_count": self.instance_count,
            "instance_type": self.instance_type,
            "volume_size": self.volume_size_gb,
            "max_run": self.max_runtime_seconds,
            "output_path": self.output_s3_uri,
            "hyperparameters": self.hyperparameters(),
            "environment": {
                "MLFLOW_TRACKING_URI": self.environment.get("MLFLOW_TRACKING_URI", ""),
                "HF_HOME": "/opt/ml/input/data/cache",
                "TOKENIZERS_PARALLELISM": "false",
                **self.environment,
            },
            "entry_point": "throughline.training.lora",
        }
        if self.use_spot:
            kwargs.update(
                {
                    "use_spot_instances": True,
                    "max_wait": self.max_wait_seconds,
                    "checkpoint_s3_uri": f"{self.output_s3_uri.rstrip('/')}/checkpoints",
                }
            )
        return Estimator(**kwargs)

    def launch(self, *, wait: bool = False) -> str:
        """Submit the job. Returns its name."""
        estimator = self.estimator()
        LOGGER.info(
            "Launching %s on %s x%s (spot=%s)",
            self.training.run_name,
            self.instance_type,
            self.instance_count,
            self.use_spot,
        )
        estimator.fit({"train": self.train_s3_uri}, wait=wait)
        return estimator.latest_training_job.name


@dataclass
class EndpointDeployment:
    """Deploy a fine-tuned adapter behind a SageMaker endpoint.

    ``async_inference`` is the right default for corpus processing: an async endpoint
    scales to zero between batches and accepts far larger payloads than a real-time
    one, which matters when a request carries four page images.
    """

    model_data_s3_uri: str
    role_arn: str
    endpoint_name: str
    region: str = "us-east-1"
    instance_type: str = "ml.g5.2xlarge"
    instance_count: int = 1
    image_uri: str = DEFAULT_IMAGE
    async_inference: bool = True
    async_output_s3_uri: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def deploy(self) -> str:
        """Create or update the endpoint. Returns its name."""
        try:
            from sagemaker.async_inference import AsyncInferenceConfig
            from sagemaker.model import Model
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError(
                "Deployment needs the sagemaker SDK. "
                "Install with: pip install 'throughline[aws]'"
            ) from exc

        model = Model(
            image_uri=self.image_uri,
            model_data=self.model_data_s3_uri,
            role=self.role_arn,
            env={
                "SM_NUM_GPUS": "1",
                "MAX_INPUT_LENGTH": "16384",
                "MAX_TOTAL_TOKENS": "20480",
                **self.environment,
            },
        )

        kwargs: dict[str, Any] = {
            "initial_instance_count": self.instance_count,
            "instance_type": self.instance_type,
            "endpoint_name": self.endpoint_name,
        }
        if self.async_inference:
            if not self.async_output_s3_uri:
                raise ValueError("async_inference requires async_output_s3_uri.")
            kwargs["async_inference_config"] = AsyncInferenceConfig(
                output_path=self.async_output_s3_uri
            )

        model.deploy(**kwargs)
        LOGGER.info("Endpoint %s deployed", self.endpoint_name)
        return self.endpoint_name
