"""Cross-page state: merge policy, table accumulation, carry-over rendering."""

from __future__ import annotations

import pytest

from throughline.schema import registry
from throughline.schema.spec import (
    ExtractionSchema,
    FieldSpec,
    TableSpec,
)
from throughline.state.cross_page import CrossPageState, EvidenceRef

INVOICE = registry.get("invoice")


def ref(page: int, block: str = "b1", confidence: float = 0.9) -> EvidenceRef:
    return EvidenceRef(page_number=page, block_id=block, quote="q", confidence=confidence)


@pytest.fixture
def state() -> CrossPageState:
    return CrossPageState(schema=INVOICE, document_id="doc")


class TestFieldMerge:
    def test_first_observation_is_taken(self, state: CrossPageState) -> None:
        assert state.update_field("invoice_number", "INV-1", group_index=0, confidence=0.8)
        assert state.to_record()["invoice_number"] == "INV-1"

    def test_same_value_again_corroborates_rather_than_revises(self, state: CrossPageState) -> None:
        state.update_field("invoice_number", "INV-1", group_index=0, confidence=0.5)
        changed = state.update_field(
            "invoice_number", "INV-1", group_index=1, confidence=0.9, evidence=[ref(3)]
        )
        assert changed is False
        entry = state.fields["invoice_number"]
        assert entry.revision_count == 0
        assert entry.confidence == 0.9
        assert entry.evidence == [ref(3)]

    def test_higher_confidence_wins(self, state: CrossPageState) -> None:
        state.update_field("bill_to", "Wrong Co", group_index=0, confidence=0.3)
        assert state.update_field("bill_to", "Right Co", group_index=1, confidence=0.9)
        assert state.to_record()["bill_to"] == "Right Co"
        assert state.fields["bill_to"].revision_count == 1

    def test_lower_confidence_does_not_overwrite(self, state: CrossPageState) -> None:
        state.update_field("bill_to", "Right Co", group_index=0, confidence=0.9)
        assert not state.update_field("bill_to", "Wrong Co", group_index=1, confidence=0.2)
        assert state.to_record()["bill_to"] == "Right Co"

    def test_continuing_field_prefers_the_later_reading_on_a_tie(
        self, state: CrossPageState
    ) -> None:
        """total_amount is only correct once the last continuation page is read."""
        state.update_field("total_amount", "$100.00", group_index=0, confidence=0.7)
        assert state.update_field("total_amount", "$980.00", group_index=3, confidence=0.7)
        assert state.to_record()["total_amount"] == "$980.00"

    def test_non_continuing_field_keeps_the_first_on_a_tie(self, state: CrossPageState) -> None:
        state.update_field("bill_to", "First Co", group_index=0, confidence=0.7)
        assert not state.update_field("bill_to", "Second Co", group_index=3, confidence=0.7)
        assert state.to_record()["bill_to"] == "First Co"

    def test_empty_values_are_ignored(self, state: CrossPageState) -> None:
        for empty in (None, "", [], {}):
            assert not state.update_field("bill_to", empty, group_index=0, confidence=1.0)
        assert "bill_to" not in state.to_record()

    def test_unknown_field_is_ignored(self, state: CrossPageState) -> None:
        assert not state.update_field("not_in_schema", "x", group_index=0, confidence=1.0)

    def test_revision_is_recorded_as_a_note(self, state: CrossPageState) -> None:
        state.update_field("bill_to", "A", group_index=0, confidence=0.2)
        state.update_field("bill_to", "B", group_index=1, confidence=0.9)
        assert any("bill_to" in note for note in state.notes)


class TestListFields:
    def test_list_fields_append_and_deduplicate(self) -> None:
        agreement = registry.get("service_agreement")
        state = CrossPageState(schema=agreement)

        state.update_field("parties", ["Alpha Ltd"], group_index=0, confidence=0.8)
        state.update_field("parties", ["Beta Inc"], group_index=1, confidence=0.8)
        state.update_field("parties", ["alpha  ltd"], group_index=2, confidence=0.8)

        assert state.to_record()["parties"] == ["Alpha Ltd", "Beta Inc"]

    def test_scalar_into_a_list_field_is_wrapped(self) -> None:
        state = CrossPageState(schema=registry.get("service_agreement"))
        state.update_field("parties", "Solo Corp", group_index=0, confidence=0.8)
        assert state.to_record()["parties"] == ["Solo Corp"]


class TestTableAccumulation:
    def test_rows_accumulate_across_groups(self, state: CrossPageState) -> None:
        assert state.append_rows(
            "line_items",
            [{"line_number": 1, "description": "A"}, {"line_number": 2, "description": "B"}],
            group_index=0,
        ) == 2
        assert state.append_rows(
            "line_items", [{"line_number": 3, "description": "C"}], group_index=1
        ) == 1
        assert state.row_count("line_items") == 3

    def test_overlap_repeat_is_dropped_by_row_key(self, state: CrossPageState) -> None:
        """The page overlap re-shows the boundary row; it must not double-count."""
        rows = [{"line_number": 6, "description": "Bearing", "amount": "$8.00"}]
        state.append_rows("line_items", rows, group_index=0)
        added = state.append_rows("line_items", rows, group_index=1)

        assert added == 0
        assert state.row_count("line_items") == 1

    def test_row_key_normalisation_tolerates_whitespace_and_case(
        self, state: CrossPageState
    ) -> None:
        state.append_rows("line_items", [{"line_number": 1, "description": "Wide Widget"}], group_index=0)
        added = state.append_rows(
            "line_items", [{"line_number": 1, "description": "wide  widget"}], group_index=1
        )
        assert added == 0

    def test_rows_differing_outside_the_key_are_still_one_row(
        self, state: CrossPageState
    ) -> None:
        state.append_rows("line_items", [{"line_number": 1, "description": "A", "amount": "$1.00"}], group_index=0)
        added = state.append_rows(
            "line_items", [{"line_number": 1, "description": "A", "amount": "$9.99"}], group_index=1
        )
        assert added == 0

    def test_blank_rows_are_skipped(self, state: CrossPageState) -> None:
        assert state.append_rows("line_items", [{}, {"line_number": None}], group_index=0) == 0

    def test_unknown_table_is_ignored(self, state: CrossPageState) -> None:
        assert state.append_rows("nope", [{"a": 1}], group_index=0) == 0

    def test_table_without_key_columns_falls_back_to_whole_row(self) -> None:
        schema = ExtractionSchema(
            name="keyless",
            tables=(
                TableSpec(
                    name="rows",
                    columns=(FieldSpec(name="a"), FieldSpec(name="b")),
                ),
            ),
        )
        state = CrossPageState(schema=schema)
        state.append_rows("rows", [{"a": "1", "b": "2"}], group_index=0)
        assert state.append_rows("rows", [{"a": "1", "b": "2"}], group_index=1) == 0
        assert state.append_rows("rows", [{"a": "1", "b": "3"}], group_index=1) == 1


class TestInspection:
    def test_missing_required_tracks_progress(self, state: CrossPageState) -> None:
        assert set(state.missing_required()) == set(INVOICE.required_keys)

        state.update_field("invoice_number", "INV-1", group_index=0, confidence=1.0)
        state.update_field("invoice_date", "2026-01-01", group_index=0, confidence=1.0)
        state.update_field("vendor_name", "Acme", group_index=0, confidence=1.0)
        state.update_field("total_amount", "$1.00", group_index=0, confidence=1.0)
        state.append_rows("line_items", [{"line_number": 1}], group_index=0)

        assert state.missing_required() == []

    def test_coverage_rises_as_fields_fill(self, state: CrossPageState) -> None:
        assert state.coverage() == 0.0
        state.update_field("invoice_number", "INV-1", group_index=0, confidence=1.0)
        assert 0 < state.coverage() < 1

    def test_mean_confidence(self, state: CrossPageState) -> None:
        state.update_field("invoice_number", "A", group_index=0, confidence=0.6)
        state.update_field("bill_to", "B", group_index=0, confidence=1.0)
        assert state.mean_confidence() == pytest.approx(0.8)


class TestCarryOver:
    def test_first_group_carry_over_says_so(self, state: CrossPageState) -> None:
        assert "first page group" in state.render_carry_over()

    def test_carry_over_lists_known_fields_and_missing_ones(
        self, state: CrossPageState
    ) -> None:
        state.update_field(
            "invoice_number", "INV-1", group_index=0, confidence=0.9, evidence=[ref(1, "p1b2")]
        )
        rendered = state.render_carry_over()

        assert "invoice_number" in rendered
        assert "p1:p1b2" in rendered
        assert "Still missing (required)" in rendered
        assert "total_amount" in rendered

    def test_open_table_is_announced(self, state: CrossPageState) -> None:
        state.append_rows("line_items", [{"line_number": 1}], group_index=0)
        state.mark_table_open("line_items")
        rendered = state.render_carry_over()

        assert "CONTINUING" in rendered
        assert "repeated column header is not a new row" in rendered

    def test_carry_over_stays_bounded(self, state: CrossPageState) -> None:
        """A carry-over that grows with the document defeats bounded processing."""
        state.append_rows(
            "line_items",
            [{"line_number": n, "description": "x" * 200} for n in range(400)],
            group_index=0,
        )
        for name in ("invoice_number", "bill_to", "vendor_name", "payment_terms"):
            state.update_field(name, "y" * 300, group_index=0, confidence=0.9)

        assert len(state.render_carry_over(max_chars=1_800)) <= 1_800


class TestSerialisation:
    def test_round_trip_preserves_state(self, state: CrossPageState) -> None:
        state.update_field(
            "invoice_number", "INV-1", group_index=0, confidence=0.9, evidence=[ref(1)]
        )
        state.append_rows("line_items", [{"line_number": 1, "description": "A"}], group_index=0)
        state.mark_table_open("line_items")
        state.record_group(0, [1, 2, 3])

        restored = CrossPageState.from_dict(state.to_dict(), INVOICE)

        assert restored.to_record() == state.to_record()
        assert restored.open_tables == {"line_items"}
        assert restored.pages_seen == {1, 2, 3}
        assert restored.fields["invoice_number"].evidence[0].page_number == 1
