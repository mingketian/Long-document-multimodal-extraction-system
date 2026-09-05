"""Page-group dataset construction and extraction metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from throughline.evaluation.harness import LabelledDocument, load_corpus
from throughline.evaluation.metrics import (
    KeyScore,
    MetricsReport,
    aggregate,
    score_document,
    values_match,
)
from throughline.grouping.page_groups import GroupingConfig
from throughline.schema import registry
from throughline.schema.spec import FieldSpec, FieldType
from throughline.training.dataset import build_examples, split, write_jsonl

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
INVOICE = registry.get("invoice")


@pytest.fixture(scope="module")
def labelled() -> LabelledDocument:
    return LabelledDocument.from_json_file(EXAMPLES / "corpus" / "invoice_0001.json")


@pytest.fixture(scope="module")
def row_pages() -> dict:
    payload = json.loads((EXAMPLES / "corpus" / "invoice_0001.json").read_text())
    return payload["row_pages"]


class TestValueMatching:
    @pytest.mark.parametrize(
        ("predicted", "gold", "field_type", "expected"),
        [
            ("$1,240.00", "1240.0", FieldType.CURRENCY, True),
            ("$1,240.00", "$1,240.01", FieldType.CURRENCY, False),
            ("2026-01-15", "01/15/2026", FieldType.DATE, True),
            ("15 January 2026", "2026-01-15", FieldType.DATE, True),
            ("  Acme  Ltd ", "acme ltd", FieldType.STRING, True),
            ("1,240", "1240", FieldType.INTEGER, True),
            (None, None, FieldType.STRING, True),
            (None, "x", FieldType.STRING, False),
        ],
    )
    def test_matching_is_type_aware(
        self, predicted: object, gold: object, field_type: FieldType, expected: bool
    ) -> None:
        spec = FieldSpec(name="f", type=field_type)
        assert values_match(predicted, gold, spec) is expected


class TestKeyScore:
    def test_perfect_score(self) -> None:
        score = KeyScore("k", true_positive=10)
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0
        assert score.support == 10

    def test_wrong_value_costs_both_precision_and_recall(self) -> None:
        score = KeyScore("k", true_positive=0, false_positive=1, false_negative=1)
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.f1 == 0.0

    def test_empty_score_does_not_divide_by_zero(self) -> None:
        assert KeyScore("k").f1 == 0.0


class TestScoring:
    def test_a_perfect_prediction_scores_one(self, labelled: LabelledDocument) -> None:
        report = MetricsReport()
        score_document(INVOICE, dict(labelled.gold), labelled.gold, report)

        assert report.weighted_f1 == pytest.approx(1.0)
        assert report.schema_valid_rate == 1.0

    def test_a_missing_field_lowers_recall_only(self, labelled: LabelledDocument) -> None:
        predicted = dict(labelled.gold)
        del predicted["purchase_order"]
        report = MetricsReport()
        score_document(INVOICE, predicted, labelled.gold, report)

        score = report.per_key["purchase_order"]
        assert score.false_negative == 1
        assert score.false_positive == 0

    def test_a_wrong_field_costs_precision_and_recall(self, labelled: LabelledDocument) -> None:
        predicted = {**labelled.gold, "invoice_number": "WRONG"}
        report = MetricsReport()
        score_document(INVOICE, predicted, labelled.gold, report)

        score = report.per_key["invoice_number"]
        assert score.false_positive == 1
        assert score.false_negative == 1

    def test_missing_table_rows_are_counted(self, labelled: LabelledDocument) -> None:
        predicted = {**labelled.gold, "line_items": labelled.gold["line_items"][:3]}
        report = MetricsReport()
        score_document(INVOICE, predicted, labelled.gold, report)

        score = report.per_key["line_items"]
        assert score.true_positive == 3
        assert score.false_negative == len(labelled.gold["line_items"]) - 3

    def test_cross_page_fields_are_scored_separately(self, labelled: LabelledDocument) -> None:
        """Only fields whose gold evidence spans pages count toward cross-page accuracy."""
        report = MetricsReport()
        score_document(
            INVOICE,
            dict(labelled.gold),
            labelled.gold,
            report,
            gold_evidence=labelled.gold_evidence,
        )

        assert report.cross_page_total > 0
        assert report.cross_page_accuracy == pytest.approx(1.0)

    def test_schema_invalid_prediction_lowers_the_valid_rate(
        self, labelled: LabelledDocument
    ) -> None:
        predicted = dict(labelled.gold)
        del predicted["total_amount"]
        report = MetricsReport()
        score_document(INVOICE, predicted, labelled.gold, report)

        assert report.schema_valid_rate == 0.0

    def test_weighted_f1_favours_high_support_keys(self) -> None:
        report = MetricsReport()
        report.per_key["common"] = KeyScore("common", true_positive=100)
        report.per_key["rare"] = KeyScore("rare", false_negative=1)

        assert report.weighted_f1 > 0.98
        assert report.macro_f1 == pytest.approx(0.5)

    def test_aggregate_sums_shards(self) -> None:
        left = MetricsReport(documents=2, schema_valid=2)
        left.per_key["a"] = KeyScore("a", true_positive=3)
        right = MetricsReport(documents=3, schema_valid=1)
        right.per_key["a"] = KeyScore("a", true_positive=1, false_negative=2)

        combined = aggregate([left, right])
        assert combined.documents == 5
        assert combined.schema_valid == 3
        assert combined.per_key["a"].true_positive == 4
        assert combined.per_key["a"].false_negative == 2

    def test_report_table_renders(self, labelled: LabelledDocument) -> None:
        report = MetricsReport()
        score_document(INVOICE, dict(labelled.gold), labelled.gold, report)
        rendered = report.table()

        assert "weighted F1" in rendered
        assert "invoice_number" in rendered


class TestCorpusLoading:
    def test_load_corpus_reads_every_document(self) -> None:
        corpus = load_corpus(EXAMPLES / "corpus")
        assert len(corpus) == 12
        assert all(item.gold for item in corpus)

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(NotADirectoryError):
            load_corpus(EXAMPLES / "does_not_exist")


class TestDatasetConstruction:
    def test_one_example_per_productive_group(
        self, labelled: LabelledDocument, row_pages: dict
    ) -> None:
        examples = build_examples(
            labelled.document,
            INVOICE,
            labelled.gold,
            labelled.gold_evidence,
            grouping=GroupingConfig(max_pages=2, overlap=1),
            row_pages=row_pages,
        )
        assert examples
        assert all(example.document_id == labelled.document_id for example in examples)

    def test_targets_are_valid_envelopes(
        self, labelled: LabelledDocument, row_pages: dict
    ) -> None:
        examples = build_examples(
            labelled.document, INVOICE, labelled.gold, labelled.gold_evidence,
            row_pages=row_pages,
        )
        for example in examples:
            payload = json.loads(example.target)
            assert set(payload) == {"fields", "evidence", "tables", "open_tables"}

    def test_a_field_is_taught_once_not_repeated(
        self, labelled: LabelledDocument, row_pages: dict
    ) -> None:
        """Rule 4 of the system prompt has to be demonstrated, not merely stated."""
        examples = build_examples(
            labelled.document, INVOICE, labelled.gold, labelled.gold_evidence,
            grouping=GroupingConfig(max_pages=2, overlap=1), row_pages=row_pages,
        )
        occurrences = sum(
            1 for e in examples if "invoice_number" in json.loads(e.target)["fields"]
        )
        assert occurrences == 1

    def test_open_tables_is_announced_before_the_last_group(
        self, labelled: LabelledDocument, row_pages: dict
    ) -> None:
        examples = build_examples(
            labelled.document, INVOICE, labelled.gold, labelled.gold_evidence,
            grouping=GroupingConfig(max_pages=2, overlap=1), row_pages=row_pages,
        )
        announced = [e for e in examples if json.loads(e.target)["open_tables"]]
        assert announced, "no group announced a continuing table"

    def test_prompt_carries_carry_over_after_the_first_group(
        self, labelled: LabelledDocument, row_pages: dict
    ) -> None:
        examples = build_examples(
            labelled.document, INVOICE, labelled.gold, labelled.gold_evidence,
            grouping=GroupingConfig(max_pages=2, overlap=1), row_pages=row_pages,
        )
        later = [e for e in examples if e.group_index > 0]
        assert later
        assert all("CARRY-OVER STATE" in e.user for e in later)

    def test_messages_have_the_three_chat_turns(
        self, labelled: LabelledDocument, row_pages: dict
    ) -> None:
        example = build_examples(
            labelled.document, INVOICE, labelled.gold, labelled.gold_evidence,
            row_pages=row_pages,
        )[0]
        roles = [turn["role"] for turn in example.to_messages()]
        assert roles == ["system", "user", "assistant"]

    def test_split_is_by_document_never_by_example(self) -> None:
        """Splitting by example leaks the answer across the carry-over boundary."""
        corpus = load_corpus(EXAMPLES / "corpus")
        examples = []
        for item in corpus:
            examples.extend(
                build_examples(item.document, INVOICE, item.gold, item.gold_evidence)
            )

        train, validation = split(examples, train_fraction=0.75)
        train_ids = {e.document_id for e in train}
        validation_ids = {e.document_id for e in validation}

        assert train_ids and validation_ids
        assert train_ids.isdisjoint(validation_ids)

    def test_jsonl_round_trip(
        self, labelled: LabelledDocument, row_pages: dict, tmp_path: Path
    ) -> None:
        examples = build_examples(
            labelled.document, INVOICE, labelled.gold, labelled.gold_evidence,
            row_pages=row_pages,
        )
        path = tmp_path / "train.jsonl"
        assert write_jsonl(examples, path) == len(examples)

        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == len(examples)
        assert all("messages" in line for line in lines)
