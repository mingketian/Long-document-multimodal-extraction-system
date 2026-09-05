"""Extraction metrics.

Four numbers, because extraction fails in four independent ways and a single score
hides which one is happening.

* **Weighted F1** - did we get the values right? Support-weighted across fields, so
  a schema with one frequently-present field and nine rare ones is not dominated by
  the rare ones.
* **Schema-valid rate** - is the output even usable downstream? A record that is
  90% correct but fails its contract cannot be written to a database.
* **Cross-page field accuracy** - the subset of fields whose gold evidence spans more
  than one page group. This is the metric page grouping and cross-page state exist to
  move, and it is the one that whole-document accuracy averages away.
* **Citation precision** - of the evidence pointers emitted, how many actually
  support the value. A system that is right for unverifiable reasons is not
  auditable.

Matching is type-aware: currency compares numerically so ``$1,240.00`` equals
``1240.0``; dates compare after normalisation; strings compare case- and
whitespace-insensitively. That is not leniency, it is refusing to score formatting
as if it were extraction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from throughline.schema.spec import Cardinality, ExtractionSchema, FieldSpec, FieldType
from throughline.schema.validate import validate_record

_CURRENCY_STRIP = re.compile(r"[^\d.\-]")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y")


# ── value comparison ──────────────────────────────────────────────────────
def _normalise_date(value: Any) -> str | None:
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalise_currency(value: Any) -> float | None:
    text = _CURRENCY_STRIP.sub("", str(value))
    if not text or text in {"-", "."}:
        return None
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def values_match(predicted: Any, gold: Any, spec: FieldSpec | None = None) -> bool:
    """Type-aware equality between a predicted and a gold value."""
    if predicted is None or gold is None:
        return predicted is None and gold is None

    field_type = spec.type if spec else FieldType.STRING

    if field_type is FieldType.CURRENCY:
        left, right = _normalise_currency(predicted), _normalise_currency(gold)
        return left is not None and left == right

    if field_type is FieldType.DATE:
        left, right = _normalise_date(predicted), _normalise_date(gold)
        return left is not None and left == right

    if field_type in {FieldType.NUMBER, FieldType.INTEGER}:
        try:
            return abs(float(str(predicted).replace(",", "")) - float(str(gold).replace(",", ""))) < 1e-6
        except (TypeError, ValueError):
            return False

    if field_type is FieldType.BOOLEAN:
        return bool(predicted) == bool(gold)

    return " ".join(str(predicted).lower().split()) == " ".join(str(gold).lower().split())


# ── per-key scoring ───────────────────────────────────────────────────────
@dataclass
class KeyScore:
    """Precision/recall counts for one schema key across a corpus."""

    key: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def support(self) -> int:
        """Number of gold instances; the weight in weighted F1."""
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
        }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _score_scalar(score: KeyScore, predicted: Any, gold: Any, spec: FieldSpec | None) -> None:
    has_prediction = predicted not in (None, "", [], {})
    has_gold = gold not in (None, "", [], {})

    if has_gold and has_prediction:
        if values_match(predicted, gold, spec):
            score.true_positive += 1
        else:
            score.false_positive += 1
            score.false_negative += 1
    elif has_gold:
        score.false_negative += 1
    elif has_prediction:
        score.false_positive += 1


def _score_multi(
    score: KeyScore, predicted: Any, gold: Any, spec: FieldSpec | None
) -> None:
    """Bag matching for list fields: each gold item is consumed at most once."""
    predicted_items = _as_list(predicted)
    remaining = _as_list(gold)

    for item in predicted_items:
        matched = next(
            (index for index, g in enumerate(remaining) if values_match(item, g, spec)), None
        )
        if matched is None:
            score.false_positive += 1
        else:
            score.true_positive += 1
            remaining.pop(matched)
    score.false_negative += len(remaining)


def _row_signature(row: dict[str, Any], key_columns: Sequence[str]) -> tuple[Any, ...]:
    columns = key_columns or sorted(row)
    return tuple(" ".join(str(row.get(c, "")).lower().split()) for c in columns)


def _score_table(
    score: KeyScore,
    predicted_rows: Any,
    gold_rows: Any,
    key_columns: Sequence[str],
) -> None:
    """Rows match on their key columns; unmatched rows on either side are errors."""
    predicted_list = [row for row in _as_list(predicted_rows) if isinstance(row, dict)]
    gold_list = [row for row in _as_list(gold_rows) if isinstance(row, dict)]

    remaining = [_row_signature(row, key_columns) for row in gold_list]
    for row in predicted_list:
        signature = _row_signature(row, key_columns)
        if signature in remaining:
            remaining.remove(signature)
            score.true_positive += 1
        else:
            score.false_positive += 1
    score.false_negative += len(remaining)


# ── corpus-level report ───────────────────────────────────────────────────
@dataclass
class MetricsReport:
    """Aggregate metrics over a corpus."""

    per_key: dict[str, KeyScore] = field(default_factory=dict)
    documents: int = 0
    schema_valid: int = 0
    cross_page_correct: int = 0
    cross_page_total: int = 0
    citations_emitted: int = 0
    citations_verified: int = 0
    wall_seconds: float = 0.0
    pages_read: int = 0
    pages_total: int = 0

    # ── headline numbers ──────────────────────────────────────────────
    @property
    def weighted_f1(self) -> float:
        """Support-weighted mean F1 across keys."""
        total_support = sum(score.support for score in self.per_key.values())
        if not total_support:
            return 0.0
        return sum(
            score.f1 * score.support for score in self.per_key.values()
        ) / total_support

    @property
    def macro_f1(self) -> float:
        scores = [score.f1 for score in self.per_key.values() if score.support]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def micro_f1(self) -> float:
        tp = sum(s.true_positive for s in self.per_key.values())
        fp = sum(s.false_positive for s in self.per_key.values())
        fn = sum(s.false_negative for s in self.per_key.values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    @property
    def schema_valid_rate(self) -> float:
        return self.schema_valid / self.documents if self.documents else 0.0

    @property
    def cross_page_accuracy(self) -> float:
        """Accuracy restricted to fields whose evidence spans page groups."""
        return (
            self.cross_page_correct / self.cross_page_total if self.cross_page_total else 0.0
        )

    @property
    def citation_precision(self) -> float:
        return (
            self.citations_verified / self.citations_emitted if self.citations_emitted else 0.0
        )

    @property
    def pages_read_fraction(self) -> float:
        return self.pages_read / self.pages_total if self.pages_total else 0.0

    @property
    def seconds_per_document(self) -> float:
        return self.wall_seconds / self.documents if self.documents else 0.0

    def headline(self) -> dict[str, float]:
        """The five numbers worth putting on a dashboard."""
        return {
            "weighted_f1": round(self.weighted_f1, 4),
            "schema_valid_rate": round(self.schema_valid_rate, 4),
            "cross_page_accuracy": round(self.cross_page_accuracy, 4),
            "citation_precision": round(self.citation_precision, 4),
            "seconds_per_document": round(self.seconds_per_document, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.headline(),
            "macro_f1": round(self.macro_f1, 4),
            "micro_f1": round(self.micro_f1, 4),
            "documents": self.documents,
            "pages_read": self.pages_read,
            "pages_total": self.pages_total,
            "pages_read_fraction": round(self.pages_read_fraction, 4),
            "wall_seconds": round(self.wall_seconds, 3),
            "per_key": {key: score.to_dict() for key, score in sorted(self.per_key.items())},
        }

    def table(self) -> str:
        """Per-key breakdown as a fixed-width table for terminals and logs."""
        rows = sorted(self.per_key.values(), key=lambda s: (-s.support, s.key))
        width = max((len(row.key) for row in rows), default=3) + 2
        lines = [f"{'key':<{width}}{'P':>8}{'R':>8}{'F1':>8}{'n':>7}"]
        lines.append("-" * (width + 31))
        for row in rows:
            lines.append(
                f"{row.key:<{width}}{row.precision:>8.3f}{row.recall:>8.3f}"
                f"{row.f1:>8.3f}{row.support:>7d}"
            )
        lines.append("-" * (width + 31))
        lines.append(f"{'weighted F1':<{width}}{self.weighted_f1:>24.3f}")
        return "\n".join(lines)


def _gold_pages(evidence: Any) -> set[int]:
    pages: set[int] = set()
    for entry in _as_list(evidence):
        if isinstance(entry, dict) and entry.get("page_number") is not None:
            pages.add(int(entry["page_number"]))
        elif isinstance(entry, int):
            pages.add(entry)
    return pages


def score_document(
    schema: ExtractionSchema,
    predicted: dict[str, Any],
    gold: dict[str, Any],
    report: MetricsReport,
    *,
    gold_evidence: dict[str, Any] | None = None,
    predicted_evidence: dict[str, Any] | None = None,
    cross_page_threshold: int = 2,
) -> None:
    """Fold one document's result into a running :class:`MetricsReport`."""
    report.documents += 1

    _, validation = validate_record(schema, dict(predicted))
    if validation.is_valid:
        report.schema_valid += 1

    for spec in schema.fields:
        score = report.per_key.setdefault(spec.name, KeyScore(spec.name))
        prediction = predicted.get(spec.name)
        truth = gold.get(spec.name)

        if spec.cardinality is Cardinality.MANY:
            _score_multi(score, prediction, truth, spec)
        else:
            _score_scalar(score, prediction, truth, spec)

        # A field is "cross-page" when its gold evidence spans several pages.
        if gold_evidence:
            pages = _gold_pages(gold_evidence.get(spec.name))
            if len(pages) >= cross_page_threshold:
                report.cross_page_total += 1
                if values_match(prediction, truth, spec):
                    report.cross_page_correct += 1

    for table in schema.tables:
        score = report.per_key.setdefault(table.name, KeyScore(table.name))
        _score_table(
            score, predicted.get(table.name), gold.get(table.name), table.row_key_columns
        )
        if gold_evidence:
            pages = _gold_pages(gold_evidence.get(table.name))
            if len(pages) >= cross_page_threshold:
                report.cross_page_total += 1
                gold_rows = _as_list(gold.get(table.name))
                predicted_rows = _as_list(predicted.get(table.name))
                if len(gold_rows) == len(predicted_rows) and score.false_negative == 0:
                    report.cross_page_correct += 1


def aggregate(reports: Iterable[MetricsReport]) -> MetricsReport:
    """Combine per-shard reports into one, for distributed evaluation."""
    combined = MetricsReport()
    for report in reports:
        combined.documents += report.documents
        combined.schema_valid += report.schema_valid
        combined.cross_page_correct += report.cross_page_correct
        combined.cross_page_total += report.cross_page_total
        combined.citations_emitted += report.citations_emitted
        combined.citations_verified += report.citations_verified
        combined.wall_seconds += report.wall_seconds
        combined.pages_read += report.pages_read
        combined.pages_total += report.pages_total
        for key, score in report.per_key.items():
            target = combined.per_key.setdefault(key, KeyScore(key))
            target.true_positive += score.true_positive
            target.false_positive += score.false_positive
            target.false_negative += score.false_negative
    return combined
