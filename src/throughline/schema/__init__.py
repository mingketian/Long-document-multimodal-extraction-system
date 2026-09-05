"""Extraction schemas: the contract that drives the whole pipeline."""

from throughline.schema.spec import (
    Cardinality,
    ExtractionSchema,
    FieldSpec,
    FieldType,
    TableSpec,
)
from throughline.schema.validate import (
    ValidationReport,
    Violation,
    is_schema_valid,
    validate_record,
)

__all__ = [
    "Cardinality",
    "ExtractionSchema",
    "FieldSpec",
    "FieldType",
    "TableSpec",
    "ValidationReport",
    "Violation",
    "is_schema_valid",
    "validate_record",
]
