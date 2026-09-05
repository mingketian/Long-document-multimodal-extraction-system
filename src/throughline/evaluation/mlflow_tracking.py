"""MLflow tracking.

Every evaluation run and every training job logs here, which is what makes "the LoRA
run beat the base model" a checkable claim rather than a remembered one. Params
capture the configuration, metrics capture the outcome, and the per-key F1 breakdown
is logged alongside the headline number so a regression can be traced to the field
that caused it.

MLflow is an optional dependency. When it is absent - CI, a laptop, a quick local
sweep - every function here degrades to a structured log line rather than failing.
An evaluation harness that cannot run without a tracking server is a tracking server
with an evaluation harness attached.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from throughline.evaluation.harness import EvaluationRun

LOGGER = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "throughline"


def is_available() -> bool:
    """True when mlflow is importable."""
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def tracking_uri() -> str | None:
    """Configured tracking URI, or ``None`` for MLflow's local default."""
    return os.environ.get("MLFLOW_TRACKING_URI")


@contextmanager
def run_context(
    run_name: str,
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Open an MLflow run, or a no-op context when MLflow is unavailable.

    Usage::

        with run_context("lora-r16") as run:
            log_params({...})
            log_metrics({...})
    """
    if not is_available():
        LOGGER.info("MLflow not installed; run %r will only be logged locally.", run_name)
        yield None
        return

    import mlflow

    uri = tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name, nested=nested) as active:
        if tags:
            mlflow.set_tags(tags)
        yield active


def log_params(params: dict[str, Any]) -> None:
    """Log configuration. Values are stringified; MLflow params are text."""
    if not params:
        return
    if not is_available():
        LOGGER.info("params: %s", params)
        return

    import mlflow

    if mlflow.active_run() is None:
        LOGGER.info("params (no active run): %s", params)
        return
    # MLflow rejects params over 500 chars; truncate rather than fail the run.
    mlflow.log_params({key: str(value)[:500] for key, value in params.items()})


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    """Log numeric metrics, skipping anything non-numeric."""
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not numeric:
        return
    if not is_available():
        LOGGER.info("metrics%s: %s", f" (step {step})" if step is not None else "", numeric)
        return

    import mlflow

    if mlflow.active_run() is None:
        LOGGER.info("metrics (no active run): %s", numeric)
        return
    mlflow.log_metrics(numeric, step=step)


def log_artifact(path: str, *, artifact_path: str | None = None) -> None:
    """Attach a file to the active run."""
    if not is_available():
        LOGGER.info("artifact: %s", path)
        return

    import mlflow

    if mlflow.active_run() is None:
        LOGGER.info("artifact (no active run): %s", path)
        return
    mlflow.log_artifact(path, artifact_path=artifact_path)


def log_dict(payload: dict[str, Any], filename: str) -> None:
    """Attach a JSON blob to the active run."""
    if not is_available():
        LOGGER.debug("dict artifact %s: %s keys", filename, len(payload))
        return

    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_dict(payload, filename)


def log_run(run: EvaluationRun) -> None:
    """Log one complete evaluation run: params, headline metrics, per-key F1.

    Per-key F1 is logged as its own metric (``f1/<key>``) so a dashboard can chart
    the field that regressed, not just the average that moved.
    """
    with run_context(run.config.run_name, experiment=run.config.experiment, tags=run.config.tags):
        log_params(
            {
                **run.config.params,
                "documents": run.report.documents,
                "pages_total": run.report.pages_total,
            }
        )

        report = run.report
        log_metrics(
            {
                **report.headline(),
                "macro_f1": report.macro_f1,
                "micro_f1": report.micro_f1,
                "pages_read_fraction": report.pages_read_fraction,
                "wall_seconds": report.wall_seconds,
            }
        )
        log_metrics({f"f1/{key}": score.f1 for key, score in report.per_key.items()})
        log_metrics(
            {f"support/{key}": float(score.support) for key, score in report.per_key.items()}
        )
        log_dict(run.to_dict(), "run.json")

    LOGGER.info("%s", run.summary())
