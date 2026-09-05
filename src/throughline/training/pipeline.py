"""The retraining pipeline, as a SageMaker Pipelines definition.

Everything needed to retrain exists as separate pieces — a dataset builder, a LoRA
trainer, an evaluation harness, a promotion gate, a deployment helper. Leaving them
separate means retraining is a runbook someone follows, which means it happens rarely,
inconsistently, and differently each time.

This module wires them into one DAG with a **conditional edge**: the deployment steps
execute only if the promotion gate passed. That conditional is the whole point. A
pipeline that always deploys what it trained is a way to ship regressions on a
schedule.

```
   BuildDataset ──▶ Train ──▶ Evaluate ──▶ ⟨gate passed?⟩ ──┬─ yes ─▶ Register ──▶ Deploy
                                                            └─ no  ─▶ Fail (with the report)
```

The steps run as SageMaker Processing and Training jobs, so each is independently
retryable, independently sized, and logged where the rest of the platform's telemetry
already lives. Building the definition needs no AWS calls — :func:`build_pipeline`
returns an object you can inspect or diff in CI before anything is submitted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from throughline.training.lora import TrainingConfig
from throughline.training.registry import PromotionGate

LOGGER = logging.getLogger(__name__)

DEFAULT_TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training:2.3.0-transformers4.46.1-gpu-py311-cu121-ubuntu20.04"
)
DEFAULT_PROCESSING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.3.0-transformers4.46.1-cpu-py311-ubuntu22.04"
)


@dataclass
class PipelineConfig:
    """Everything the retraining DAG needs to know.

    Args:
        pipeline_name: Name in SageMaker Pipelines. Versioned by execution, not by name.
        role_arn: Execution role for every step.
        bucket: S3 bucket for inputs, artefacts and outputs.
        prefix: Key prefix inside the bucket.
        schema: Extraction schema being trained for. One pipeline per schema — a
            single pipeline for all of them would make the promotion gate compare
            models that were never alternatives.
        training: LoRA hyperparameters, passed through as job hyperparameters.
        gate: The promotion gate, serialised into the evaluation step so the
            threshold that made a decision is recorded with the decision.
        endpoint_name: Endpoint updated on promotion.
        training_instance / processing_instance: Instance types per step class.
        use_spot: Spot for training. Safe because the trainer checkpoints.
    """

    pipeline_name: str = "throughline-retrain"
    role_arn: str = ""
    bucket: str = ""
    prefix: str = "throughline"
    schema: str = "invoice"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gate: PromotionGate = field(default_factory=PromotionGate)
    endpoint_name: str = "throughline-qwen25vl"
    training_instance: str = "ml.g5.2xlarge"
    processing_instance: str = "ml.m5.2xlarge"
    evaluation_instance: str = "ml.g5.xlarge"
    use_spot: bool = True
    max_runtime_seconds: int = 24 * 3600
    training_image: str = DEFAULT_TRAINING_IMAGE
    processing_image: str = DEFAULT_PROCESSING_IMAGE

    def s3(self, *parts: str) -> str:
        """Build an ``s3://bucket/prefix/...`` URI."""
        if not self.bucket:
            raise ValueError("PipelineConfig.bucket is required to build S3 URIs.")
        tail = "/".join(part.strip("/") for part in parts if part)
        return f"s3://{self.bucket}/{self.prefix.strip('/')}/{tail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "schema": self.schema,
            "endpoint_name": self.endpoint_name,
            "training_instance": self.training_instance,
            "processing_instance": self.processing_instance,
            "evaluation_instance": self.evaluation_instance,
            "use_spot": self.use_spot,
            "training": self.training.to_dict(),
            "gate": {
                "primary_metric": self.gate.primary_metric,
                "min_improvement": self.gate.min_improvement,
                "guarded_metrics": list(self.gate.guarded_metrics),
                "metric_tolerance": self.gate.metric_tolerance,
                "per_key_tolerance": self.gate.per_key_tolerance,
                "absolute_floors": dict(self.gate.absolute_floors),
                "min_corpus_size": self.gate.min_corpus_size,
            },
        }


# ── a backend-agnostic plan ───────────────────────────────────────────────
@dataclass(frozen=True)
class StepPlan:
    """One step, described without reference to any SDK.

    The plan is built first and rendered into SageMaker objects second, so the DAG
    can be inspected, diffed and unit-tested without importing the SageMaker SDK or
    holding AWS credentials. That separation is what lets CI check the pipeline
    shape on every commit.
    """

    name: str
    kind: str
    """``processing`` | ``training`` | ``condition`` | ``fail``"""

    command: list[str]
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    instance_type: str = ""
    depends_on: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "command": self.command,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "instance_type": self.instance_type,
            "depends_on": list(self.depends_on),
            "description": self.description,
        }


def build_plan(config: PipelineConfig) -> list[StepPlan]:
    """Describe the retraining DAG. Pure — makes no AWS calls."""
    return [
        StepPlan(
            name="BuildDataset",
            kind="processing",
            instance_type=config.processing_instance,
            command=[
                "throughline", "build-dataset", "/opt/ml/processing/input/corpus",
                "--schema", config.schema,
                "--out", "/opt/ml/processing/output/train.jsonl",
                "--max-pages", str(config.training.max_seq_length // 2048),
                "--train-fraction", "0.9",
            ],
            inputs={"corpus": config.s3("corpus", config.schema)},
            outputs={"dataset": config.s3("datasets", config.schema)},
            description=(
                "Replay the real pipeline's page grouping over the labelled corpus and "
                "emit one training example per page-group set. Split by document."
            ),
        ),
        StepPlan(
            name="TrainLoRA",
            kind="training",
            instance_type=config.training_instance,
            command=["python", "-m", "throughline.training.lora"],
            inputs={"train": config.s3("datasets", config.schema)},
            outputs={"adapter": config.s3("adapters", config.schema)},
            depends_on=("BuildDataset",),
            description=(
                f"LoRA r={config.training.lora.r} on {config.training.model_id}, vision "
                "tower frozen, loss on the completion only."
            ),
        ),
        StepPlan(
            name="Evaluate",
            kind="processing",
            instance_type=config.evaluation_instance,
            command=[
                "throughline", "evaluate", "/opt/ml/processing/input/holdout",
                "--schema", config.schema,
                "--profile", "accuracy",
                "--adapter", "/opt/ml/processing/input/adapter",
                "--artifacts", "/opt/ml/processing/output",
            ],
            inputs={
                "holdout": config.s3("holdout", config.schema),
                "adapter": config.s3("adapters", config.schema),
            },
            outputs={"metrics": config.s3("evaluations", config.schema)},
            depends_on=("TrainLoRA",),
            description=(
                "Score the candidate on a held-out corpus with early exit disabled, so "
                "the number is the accuracy ceiling rather than a latency configuration."
            ),
        ),
        StepPlan(
            name="PromotionGate",
            kind="condition",
            command=[
                "python", "-m", "throughline.training.registry",
                "--evaluate-gate",
                "--metrics", "/opt/ml/processing/input/metrics/run.json",
            ],
            inputs={"metrics": config.s3("evaluations", config.schema)},
            depends_on=("Evaluate",),
            description=(
                f"Promote only if {config.gate.primary_metric} improves by "
                f"≥ {config.gate.min_improvement}, no guarded metric regresses beyond "
                f"{config.gate.metric_tolerance}, and no individual field's F1 drops "
                f"more than {config.gate.per_key_tolerance}."
            ),
        ),
        StepPlan(
            name="RegisterModel",
            kind="processing",
            instance_type=config.processing_instance,
            command=["python", "-m", "throughline.training.registry", "--register"],
            inputs={
                "adapter": config.s3("adapters", config.schema),
                "metrics": config.s3("evaluations", config.schema),
            },
            outputs={"registry": config.s3("registry")},
            depends_on=("PromotionGate",),
            description="Write the model card and mark the candidate champion.",
        ),
        StepPlan(
            name="DeployEndpoint",
            kind="processing",
            instance_type=config.processing_instance,
            command=[
                "python", "-m", "throughline.training.sagemaker_launch",
                "--deploy", "--endpoint", config.endpoint_name,
            ],
            inputs={"adapter": config.s3("adapters", config.schema)},
            depends_on=("RegisterModel",),
            description=(
                f"Update {config.endpoint_name} in place. The previous adapter stays "
                "archived in the registry as the rollback target."
            ),
        ),
        StepPlan(
            name="GateFailed",
            kind="fail",
            command=[],
            depends_on=("PromotionGate",),
            description=(
                "Stop with the gate report attached. Not deploying a candidate that did "
                "not clear the gate is the pipeline succeeding at its job."
            ),
        ),
    ]


def render_plan(config: PipelineConfig) -> str:
    """The DAG as text, for logs, PR descriptions and CI output."""
    plan = build_plan(config)
    lines = [f"pipeline: {config.pipeline_name} (schema={config.schema})", ""]
    for step in plan:
        arrow = f"  ← {', '.join(step.depends_on)}" if step.depends_on else ""
        lines.append(f"{step.name} [{step.kind}]{arrow}")
        if step.instance_type:
            lines.append(f"    instance: {step.instance_type}")
        if step.description:
            lines.append(f"    {step.description}")
    return "\n".join(lines)


def validate_plan(config: PipelineConfig) -> list[str]:
    """Check the DAG is well-formed. Returns problems; empty means valid."""
    plan = build_plan(config)
    names = {step.name for step in plan}
    problems: list[str] = []

    for step in plan:
        for dependency in step.depends_on:
            if dependency not in names:
                problems.append(f"{step.name} depends on unknown step {dependency!r}")

    # Every step except the entry point must be reachable from BuildDataset.
    reachable = {"BuildDataset"}
    for _ in range(len(plan)):
        for step in plan:
            if step.depends_on and set(step.depends_on) & reachable:
                reachable.add(step.name)
    unreachable = names - reachable
    if unreachable:
        problems.append(f"unreachable steps: {sorted(unreachable)}")

    conditions = [step for step in plan if step.kind == "condition"]
    if not conditions:
        problems.append("no condition step: this pipeline would deploy unconditionally")

    return problems


# ── SageMaker rendering ───────────────────────────────────────────────────
def build_pipeline(config: PipelineConfig) -> Any:
    """Render the plan as a SageMaker ``Pipeline``.

    Requires the ``sagemaker`` SDK, but makes no AWS calls itself — the returned
    object is submitted with ``.upsert(role_arn=...)`` and started with ``.start()``.
    """
    problems = validate_plan(config)
    if problems:
        raise ValueError("Pipeline plan is invalid:\n  " + "\n  ".join(problems))
    if not config.role_arn or not config.bucket:
        raise ValueError("PipelineConfig requires both role_arn and bucket.")

    try:
        from sagemaker.estimator import Estimator
        from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
        from sagemaker.workflow.condition_step import ConditionStep
        from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
        from sagemaker.workflow.fail_step import FailStep
        from sagemaker.workflow.functions import JsonGet
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.properties import PropertyFile
        from sagemaker.workflow.steps import ProcessingStep, TrainingStep
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError(
            "Building a SageMaker pipeline needs the sagemaker SDK. "
            "Install with: pip install 'throughline[aws]'"
        ) from exc

    def processor(instance_type: str) -> Any:
        return ScriptProcessor(
            image_uri=config.processing_image,
            command=["python3"],
            role=config.role_arn,
            instance_count=1,
            instance_type=instance_type,
        )

    dataset_step = ProcessingStep(
        name="BuildDataset",
        processor=processor(config.processing_instance),
        inputs=[
            ProcessingInput(
                source=config.s3("corpus", config.schema),
                destination="/opt/ml/processing/input/corpus",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="dataset",
                source="/opt/ml/processing/output",
                destination=config.s3("datasets", config.schema),
            )
        ],
        code="tools/pipeline_steps/build_dataset.py",
    )

    estimator = Estimator(
        image_uri=config.training_image,
        role=config.role_arn,
        instance_count=1,
        instance_type=config.training_instance,
        max_run=config.max_runtime_seconds,
        output_path=config.s3("adapters", config.schema),
        hyperparameters={k: str(v) for k, v in config.training.to_dict().items()},
        use_spot_instances=config.use_spot,
        max_wait=config.max_runtime_seconds + 12 * 3600 if config.use_spot else None,
        checkpoint_s3_uri=config.s3("checkpoints", config.schema) if config.use_spot else None,
        entry_point="throughline.training.lora",
    )
    train_step = TrainingStep(
        name="TrainLoRA",
        estimator=estimator,
        inputs={
            "train": dataset_step.properties.ProcessingOutputConfig.Outputs[
                "dataset"
            ].S3Output.S3Uri
        },
    )

    metrics_file = PropertyFile(
        name="EvaluationReport", output_name="metrics", path="run.json"
    )
    evaluate_step = ProcessingStep(
        name="Evaluate",
        processor=processor(config.evaluation_instance),
        inputs=[
            ProcessingInput(
                source=config.s3("holdout", config.schema),
                destination="/opt/ml/processing/input/holdout",
            ),
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/adapter",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics",
                source="/opt/ml/processing/output",
                destination=config.s3("evaluations", config.schema),
            )
        ],
        code="tools/pipeline_steps/evaluate.py",
        property_files=[metrics_file],
    )

    register_step = ProcessingStep(
        name="RegisterModel",
        processor=processor(config.processing_instance),
        inputs=[
            ProcessingInput(
                source=config.s3("evaluations", config.schema),
                destination="/opt/ml/processing/input/metrics",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="registry",
                source="/opt/ml/processing/output",
                destination=config.s3("registry"),
            )
        ],
        code="tools/pipeline_steps/register.py",
    )
    deploy_step = ProcessingStep(
        name="DeployEndpoint",
        processor=processor(config.processing_instance),
        code="tools/pipeline_steps/deploy.py",
        depends_on=[register_step.name],
    )

    # The gate itself runs inside the evaluation step, which writes `gate.promote`
    # into run.json. The condition reads that verdict rather than re-deriving it,
    # so the deployed decision and the recorded decision cannot diverge.
    gate_step = ConditionStep(
        name="PromotionGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluate_step.name,
                    property_file=metrics_file,
                    json_path="gate.promote_score",
                ),
                right=1.0,
            )
        ],
        if_steps=[register_step, deploy_step],
        else_steps=[
            FailStep(
                name="GateFailed",
                error_message=(
                    "Promotion gate held the candidate. See gate.checks in run.json."
                ),
            )
        ],
    )

    return Pipeline(
        name=config.pipeline_name,
        steps=[dataset_step, train_step, evaluate_step, gate_step],
        parameters=[],
    )


def export_definition(config: PipelineConfig, path: str) -> str:
    """Write the plan as JSON, so a DAG change shows up as a reviewable diff."""
    payload = {
        "config": config.to_dict(),
        "steps": [step.to_dict() for step in build_plan(config)],
        "problems": validate_plan(config),
    }
    body = json.dumps(payload, indent=2, default=str)
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body + "\n", encoding="utf-8")
    return str(destination)
