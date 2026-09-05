"""Produce the result tables in results/tables/.

Two kinds of number live here, and they are kept in separate files because they mean
different things:

* ``deployment_*.csv`` - **reported outcomes** from the Ricoh USA internship
  deployment (Sep-Dec 2025), measured on ~600 proprietary enterprise documents that
  are not in this repository. They are recorded here for provenance. **They are not
  reproducible from this code** and nothing in this repo produced them.

* ``measured_*.csv`` - produced *by this repository*, right now, by running the
  pipeline over the synthetic corpus in ``examples/``. Regenerate with
  ``make results``. These are small and use the rule-based baseline backend, so the
  absolute values are low; what they demonstrate is that the accuracy/coverage
  trade-off the system is built around is real and measurable.

Run with:  python tools/make_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from throughline.config import PROFILES
from throughline.evaluation.harness import EvaluationConfig, evaluate, load_corpus
from throughline.schema import registry

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

# ── reported deployment outcomes (NOT reproducible here) ──────────────────
DEPLOYMENT_FINE_TUNING = [
    ("weighted_f1", "Weighted F1", 0.867, 0.923),
    ("schema_valid_rate", "Schema-valid output rate", 0.934, 0.986),
]

DEPLOYMENT_ORCHESTRATION = [
    ("cross_page_accuracy", "Cross-page field accuracy", 0.789, 0.887),
    ("citation_precision", "Citation precision", 0.879, 0.942),
]

DEPLOYMENT_LATENCY = [
    ("processing_time_reduction", "Processing-time reduction", 0.202),
    ("f1_delta_vs_peak", "Weighted-F1 gap vs peak accuracy", -0.003),
]

DEPLOYMENT_SCALE = [
    ("documents", "Enterprise documents processed", 600),
    ("pages", "Pages processed", 3200),
    ("labelled_page_group_sets", "Labelled page-group sets used for LoRA", 2400),
]


def write_csv(name: str, header: list[str], rows: list[tuple]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {TABLES / name}")


def write_deployment_tables() -> None:
    write_csv(
        "deployment_fine_tuning.csv",
        ["metric", "label", "before_lora", "after_lora"],
        DEPLOYMENT_FINE_TUNING,
    )
    write_csv(
        "deployment_orchestration.csv",
        ["metric", "label", "before_harness", "after_harness"],
        DEPLOYMENT_ORCHESTRATION,
    )
    write_csv("deployment_latency.csv", ["metric", "label", "value"], DEPLOYMENT_LATENCY)
    write_csv("deployment_scale.csv", ["metric", "label", "value"], DEPLOYMENT_SCALE)


# ── measured in this repository ───────────────────────────────────────────
def measure(corpus_dir: Path, schema_name: str) -> list[dict[str, Any]]:
    """Run every profile over one corpus and collect the headline metrics."""
    corpus = load_corpus(corpus_dir)
    schema = registry.get(schema_name)
    rows: list[dict[str, Any]] = []

    for profile_name, config in PROFILES.items():
        config.schema = schema_name
        config.cache.enabled = False
        run = evaluate(
            config.build_pipeline(),
            corpus,
            schema,
            EvaluationConfig(
                run_name=f"{schema_name}-{profile_name}",
                params=config.flat_params(),
                log_to_mlflow=False,
            ),
        )
        report = run.report
        rows.append(
            {
                "schema": schema_name,
                "profile": profile_name,
                "documents": report.documents,
                "weighted_f1": round(report.weighted_f1, 4),
                "schema_valid_rate": round(report.schema_valid_rate, 4),
                "citation_precision": round(report.citation_precision, 4),
                "cross_page_accuracy": round(report.cross_page_accuracy, 4),
                "pages_read_fraction": round(report.pages_read_fraction, 4),
                "pages_read": report.pages_read,
                "pages_total": report.pages_total,
                "seconds_per_document": round(report.seconds_per_document, 5),
            }
        )
        print(f"  {run.summary()}")
    return rows


def write_measured_tables() -> None:
    rows: list[dict[str, Any]] = []
    print("measuring invoice corpus (table-bearing, early exit cannot fire):")
    rows += measure(ROOT / "examples" / "corpus", "invoice")
    print("measuring agreement corpus (long, no required table):")
    rows += measure(ROOT / "examples" / "corpus_agreements", "service_agreement")

    header = list(rows[0])
    write_csv("measured_profiles.csv", header, [tuple(row[k] for k in header) for row in rows])
    (TABLES / "measured_profiles.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {TABLES / 'measured_profiles.json'}")


def write_corpus_stats() -> None:
    """Descriptive statistics of the synthetic corpus, so the scale is legible."""
    rows = []
    for name, directory, schema_name in (
        ("invoice", ROOT / "examples" / "corpus", "invoice"),
        ("service_agreement", ROOT / "examples" / "corpus_agreements", "service_agreement"),
    ):
        corpus = load_corpus(directory)
        pages = [item.document.page_count for item in corpus]
        schema = registry.get(schema_name)
        rows.append(
            (
                name,
                len(corpus),
                sum(pages),
                min(pages),
                max(pages),
                round(sum(pages) / len(pages), 1),
                len(schema.all_keys),
                len(schema.required_keys),
            )
        )

    write_csv(
        "corpus_stats.csv",
        [
            "schema",
            "documents",
            "pages_total",
            "pages_min",
            "pages_max",
            "pages_mean",
            "schema_keys",
            "required_keys",
        ],
        rows,
    )


def main() -> None:
    write_deployment_tables()
    write_corpus_stats()
    write_measured_tables()


if __name__ == "__main__":
    main()
