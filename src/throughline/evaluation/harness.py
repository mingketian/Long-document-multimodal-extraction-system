"""Evaluation harness.

Runs a pipeline over a labelled corpus and produces one :class:`MetricsReport`, with
every run logged to MLflow when it is available. The point of routing all evaluation
through one entry is comparability: a base checkpoint, a LoRA adapter, and a latency
configuration are all scored by the same code on the same corpus, so a difference in
the numbers is a difference in the system rather than in how it was measured.

A run is fully described by its config, which is logged as MLflow params. Two runs
with the same params on the same corpus should produce the same metrics; when they
do not, the cause is the model, and that is exactly the question worth asking.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from throughline.evaluation.metrics import MetricsReport, score_document
from throughline.ingest.layout import Document
from throughline.pipeline.orchestrator import ExtractionPipeline, ExtractionResult
from throughline.schema.spec import ExtractionSchema

LOGGER = logging.getLogger(__name__)


@dataclass
class LabelledDocument:
    """A document with its gold record and, optionally, gold evidence pages."""

    document: Document
    gold: dict[str, Any]
    gold_evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        return self.document.document_id

    @classmethod
    def from_json_file(cls, path: str | Path) -> LabelledDocument:
        """Load ``{"document": {...}, "gold": {...}, "gold_evidence": {...}}``."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            document=Document.from_dict(payload["document"]),
            gold=payload.get("gold", {}),
            gold_evidence=payload.get("gold_evidence", {}),
        )


def load_corpus(directory: str | Path, *, pattern: str = "*.json") -> list[LabelledDocument]:
    """Load every labelled document in a directory, sorted by filename."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {root}")
    return [LabelledDocument.from_json_file(path) for path in sorted(root.glob(pattern))]


@dataclass
class EvaluationConfig:
    """What a run is and how it should be recorded."""

    run_name: str = "eval"
    experiment: str = "throughline"
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    log_to_mlflow: bool = True
    artifacts_dir: str | Path | None = None
    """When set, per-document results are written here and logged as artifacts."""

    fail_fast: bool = False


@dataclass
class EvaluationRun:
    """The result of evaluating one configuration over one corpus."""

    config: EvaluationConfig
    report: MetricsReport
    results: list[ExtractionResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.config.run_name,
            "params": self.config.params,
            "tags": self.config.tags,
            "metrics": self.report.to_dict(),
            "duration_seconds": round(self.duration_seconds, 3),
            "documents": [
                {
                    "document_id": result.document_id,
                    "valid": result.is_valid,
                    "groups_processed": result.groups_processed,
                    "groups_total": result.groups_total,
                    "pages_read": result.pages_read,
                    "exit_reason": result.exit_reason.value if result.exit_reason else None,
                    "wall_seconds": round(result.wall_seconds, 4),
                }
                for result in self.results
            ],
        }

    def summary(self) -> str:
        headline = self.report.headline()
        return (
            f"{self.config.run_name}: "
            f"weighted F1 {headline['weighted_f1']:.3f} · "
            f"schema-valid {headline['schema_valid_rate']:.3f} · "
            f"cross-page {headline['cross_page_accuracy']:.3f} · "
            f"citations {headline['citation_precision']:.3f} · "
            f"{headline['seconds_per_document']:.2f}s/doc"
        )


def evaluate(
    pipeline: ExtractionPipeline,
    corpus: Sequence[LabelledDocument],
    schema: ExtractionSchema,
    config: EvaluationConfig | None = None,
    *,
    progress: Callable[[int, int, ExtractionResult], None] | None = None,
) -> EvaluationRun:
    """Score a pipeline over a labelled corpus.

    Args:
        pipeline: The configured pipeline under test.
        corpus: Labelled documents.
        schema: The extraction target; must match the pipeline's schema.
        config: Run naming, tagging and logging options.
        progress: Optional callback invoked after each document.

    Returns:
        An :class:`EvaluationRun` holding the report and every per-document result.
    """
    config = config or EvaluationConfig()
    report = MetricsReport()
    results: list[ExtractionResult] = []
    started = time.time()

    for index, labelled in enumerate(corpus, start=1):
        try:
            result = pipeline.run(labelled.document)
        except Exception:  # noqa: BLE001 - one document must not end the run
            if config.fail_fast:
                raise
            LOGGER.exception("Document %s failed during evaluation", labelled.document_id)
            report.documents += 1
            report.pages_total += labelled.document.page_count
            continue

        results.append(result)
        score_document(
            schema,
            result.record,
            labelled.gold,
            report,
            gold_evidence=labelled.gold_evidence,
        )

        report.wall_seconds += result.wall_seconds
        report.pages_read += result.pages_read
        report.pages_total += labelled.document.page_count
        report.citations_emitted += result.attribution.total_claims
        report.citations_verified += result.attribution.verified

        if progress is not None:
            progress(index, len(corpus), result)

    run = EvaluationRun(
        config=config, report=report, results=results, started_at=started, finished_at=time.time()
    )

    if config.artifacts_dir:
        _write_artifacts(run, Path(config.artifacts_dir))
    if config.log_to_mlflow:
        from throughline.evaluation.mlflow_tracking import log_run

        log_run(run)

    return run


def _write_artifacts(run: EvaluationRun, directory: Path) -> None:
    """Write the run summary and per-document records to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(
        json.dumps(run.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    per_document = directory / "documents"
    per_document.mkdir(exist_ok=True)
    for result in run.results:
        (per_document / f"{result.document_id}.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )


def compare(runs: Iterable[EvaluationRun]) -> str:
    """Render several runs as one comparison table."""
    rows = list(runs)
    if not rows:
        return "(no runs)"

    columns = [
        ("run", lambda r: r.config.run_name),
        ("F1", lambda r: f"{r.report.weighted_f1:.3f}"),
        ("valid", lambda r: f"{r.report.schema_valid_rate:.3f}"),
        ("x-page", lambda r: f"{r.report.cross_page_accuracy:.3f}"),
        ("cite", lambda r: f"{r.report.citation_precision:.3f}"),
        ("pages", lambda r: f"{r.report.pages_read_fraction:.1%}"),
        ("s/doc", lambda r: f"{r.report.seconds_per_document:.2f}"),
    ]

    widths = [
        max(len(header), max(len(str(getter(row))) for row in rows)) + 2
        for header, getter in columns
    ]
    lines = ["".join(h.ljust(w) for (h, _), w in zip(columns, widths, strict=True))]
    lines.append("-" * sum(widths))
    for row in rows:
        lines.append(
            "".join(str(getter(row)).ljust(w) for (_, getter), w in zip(columns, widths, strict=True))
        )
    return "\n".join(lines)
