"""Schema validation/repair and defensive output parsing."""

from __future__ import annotations

import json

import pytest

from throughline.decoding.constrained import ParseError, build_grammar, parse_envelope
from throughline.schema import registry
from throughline.schema.spec import (
    ExtractionSchema,
    FieldSpec,
    FieldType,
    TableSpec,
)
from throughline.schema.validate import is_schema_valid, validate_record

INVOICE = registry.get("invoice")

VALID_RECORD = {
    "invoice_number": "INV-1",
    "invoice_date": "2026-01-15",
    "vendor_name": "Acme Ltd",
    "total_amount": "$1,240.00",
    "line_items": [{"line_number": 1, "description": "Widget", "amount": "$1,240.00"}],
}


class TestFieldCoercion:
    @pytest.mark.parametrize(
        ("field_type", "raw", "expected"),
        [
            (FieldType.INTEGER, "1,240", 1240),
            (FieldType.INTEGER, "not a number", None),
            (FieldType.NUMBER, "3.5", 3.5),
            (FieldType.BOOLEAN, "yes", True),
            (FieldType.BOOLEAN, "NO", False),
            (FieldType.BOOLEAN, "maybe", None),
            (FieldType.DATE, "2026-01-15", "2026-01-15"),
            (FieldType.DATE, "15 January 2026", "15 January 2026"),
            (FieldType.DATE, "sometime", None),
            (FieldType.CURRENCY, "$1,240.00", "$1,240.00"),
            (FieldType.CURRENCY, "free", None),
        ],
    )
    def test_coerce(self, field_type: FieldType, raw: object, expected: object) -> None:
        assert FieldSpec(name="f", type=field_type).coerce(raw) == expected

    def test_enum_is_case_insensitive_but_canonicalises(self) -> None:
        spec = FieldSpec(name="c", type=FieldType.ENUM, enum_values=("USD", "EUR"))
        assert spec.coerce("usd") == "USD"
        assert spec.coerce("JPY") is None

    def test_enum_without_values_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="enum_values"):
            FieldSpec(name="c", type=FieldType.ENUM)


class TestValidation:
    def test_a_complete_record_is_valid(self) -> None:
        assert is_schema_valid(INVOICE, dict(VALID_RECORD))

    def test_missing_required_field_is_an_error(self) -> None:
        record = dict(VALID_RECORD)
        del record["total_amount"]
        _, report = validate_record(INVOICE, record)

        assert not report.is_valid
        assert any(v.code == "missing_required" for v in report.errors)

    def test_unknown_key_is_a_warning_and_is_dropped(self) -> None:
        record = {**VALID_RECORD, "surprise": "value"}
        repaired, report = validate_record(INVOICE, record)

        assert report.is_valid
        assert "surprise" not in repaired
        assert any(v.code == "unknown_key" for v in report.warnings)

    def test_bad_type_is_an_error(self) -> None:
        record = {**VALID_RECORD, "invoice_date": "not a date"}
        _, report = validate_record(INVOICE, record)

        assert not report.is_valid
        assert any(v.code == "type_mismatch" for v in report.errors)

    def test_single_element_list_around_a_scalar_is_repaired(self) -> None:
        record = {**VALID_RECORD, "invoice_number": ["INV-1"]}
        repaired, report = validate_record(INVOICE, record)

        assert report.is_valid
        assert repaired["invoice_number"] == "INV-1"
        assert "invoice_number" in report.repaired_paths

    def test_multi_element_list_for_a_scalar_is_a_cardinality_error(self) -> None:
        record = {**VALID_RECORD, "invoice_number": ["A", "B"]}
        _, report = validate_record(INVOICE, record)

        assert not report.is_valid
        assert any(v.code == "cardinality" for v in report.errors)

    def test_repair_never_invents_a_missing_required_field(self) -> None:
        """The one thing repair must never do in a grounded extraction system."""
        record = dict(VALID_RECORD)
        del record["vendor_name"]
        repaired, report = validate_record(INVOICE, record)

        assert "vendor_name" not in repaired
        assert not report.is_valid

    def test_unknown_table_column_is_dropped_with_a_warning(self) -> None:
        record = dict(VALID_RECORD)
        record["line_items"] = [{"line_number": 1, "description": "W", "bogus": "x"}]
        repaired, report = validate_record(INVOICE, record)

        assert report.is_valid
        assert "bogus" not in repaired["line_items"][0]
        assert any(v.code == "unknown_column" for v in report.warnings)

    def test_single_row_dict_is_wrapped(self) -> None:
        record = dict(VALID_RECORD)
        record["line_items"] = {"line_number": 1, "description": "W"}
        repaired, report = validate_record(INVOICE, record)

        assert report.is_valid
        assert isinstance(repaired["line_items"], list)

    def test_repair_false_leaves_the_input_alone(self) -> None:
        record = {**VALID_RECORD, "invoice_number": ["INV-1"]}
        returned, _ = validate_record(INVOICE, record, repair=False)
        assert returned["invoice_number"] == ["INV-1"]


class TestSchemaConstruction:
    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ExtractionSchema(
                name="dupe",
                fields=(FieldSpec(name="a"),),
                tables=(TableSpec(name="a", columns=(FieldSpec(name="x"),)),),
            )

    def test_row_key_must_reference_real_columns(self) -> None:
        with pytest.raises(ValueError, match="row_key_columns"):
            TableSpec(name="t", columns=(FieldSpec(name="x"),), row_key_columns=("nope",))

    def test_json_round_trip(self) -> None:
        restored = ExtractionSchema.from_dict(INVOICE.to_dict())
        assert restored.name == INVOICE.name
        assert restored.all_keys == INVOICE.all_keys
        assert restored.required_keys == INVOICE.required_keys

    def test_json_schema_marks_required_keys(self) -> None:
        rendered = INVOICE.to_json_schema()
        assert set(rendered["required"]) == set(INVOICE.required_keys)
        assert rendered["properties"]["line_items"]["type"] == "array"

    def test_grammar_constrains_the_whole_envelope(self) -> None:
        grammar = build_grammar(INVOICE)
        assert set(grammar["properties"]) == {"fields", "evidence", "tables", "open_tables"}
        assert grammar["additionalProperties"] is False


class TestEnvelopeParsing:
    def test_clean_json(self) -> None:
        envelope = parse_envelope(
            json.dumps({"fields": {"invoice_number": "INV-1"}, "evidence": {}}), INVOICE
        )
        assert envelope.fields == {"invoice_number": "INV-1"}
        assert not envelope.was_repaired

    def test_markdown_fence_is_stripped(self) -> None:
        envelope = parse_envelope('```json\n{"fields": {"bill_to": "Acme"}}\n```', INVOICE)
        assert envelope.fields == {"bill_to": "Acme"}
        assert "stripped markdown fence" in envelope.repairs

    def test_prose_around_the_object_is_ignored(self) -> None:
        envelope = parse_envelope(
            'Here you go:\n{"fields": {"bill_to": "Acme"}}\nHope that helps!', INVOICE
        )
        assert envelope.fields == {"bill_to": "Acme"}

    def test_trailing_comma_is_repaired(self) -> None:
        envelope = parse_envelope('{"fields": {"bill_to": "Acme",},}', INVOICE)
        assert envelope.fields == {"bill_to": "Acme"}
        assert "removed trailing comma" in envelope.repairs

    def test_truncated_generation_is_salvaged(self) -> None:
        """Long tabular decoding gets cut off; losing the whole group is too costly."""
        envelope = parse_envelope('{"fields": {"bill_to": "Acme", "vendor_name": "X"', INVOICE)
        assert envelope.fields["bill_to"] == "Acme"

    def test_python_literals_are_normalised(self) -> None:
        envelope = parse_envelope("{'fields': {'bill_to': 'Acme'}}", INVOICE)
        assert envelope.fields == {"bill_to": "Acme"}

    def test_bare_record_is_wrapped_into_an_envelope(self) -> None:
        envelope = parse_envelope(
            json.dumps({"invoice_number": "INV-1", "line_items": [{"line_number": 1}]}), INVOICE
        )
        assert envelope.fields == {"invoice_number": "INV-1"}
        assert envelope.tables["line_items"][0]["values"] == {"line_number": 1}
        assert "wrapped bare record in envelope" in envelope.repairs

    def test_undeclared_keys_are_dropped(self) -> None:
        envelope = parse_envelope(
            json.dumps({"fields": {"invoice_number": "A", "invented": "B"}}), INVOICE
        )
        assert "invented" not in envelope.fields

    def test_evidence_shapes_are_normalised(self) -> None:
        envelope = parse_envelope(
            json.dumps(
                {
                    "fields": {"invoice_number": "A", "bill_to": "B"},
                    "evidence": {"invoice_number": "p1b2", "bill_to": {"block_id": "p1b5"}},
                }
            ),
            INVOICE,
        )
        assert envelope.evidence["invoice_number"] == [{"block_id": "p1b2"}]
        assert envelope.evidence["bill_to"] == [{"block_id": "p1b5"}]

    def test_bare_row_list_is_accepted_for_a_table(self) -> None:
        envelope = parse_envelope(
            json.dumps({"fields": {}, "tables": {"line_items": [{"line_number": 1}]}}), INVOICE
        )
        assert envelope.tables["line_items"][0]["values"] == {"line_number": 1}

    def test_no_json_at_all_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_envelope("I could not find anything on these pages.", INVOICE)

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_envelope("[1, 2, 3]", INVOICE)
