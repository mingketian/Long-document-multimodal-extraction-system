"""Schema validation and repair.

Two things happen here, and keeping them separate matters:

* **Validation** answers "is this record schema-valid?" - the metric the pipeline
  reports as ``schema_valid_rate`` and the signal the early-exit policy reads.
* **Repair** makes a best-effort fix to a nearly-valid record (a number written as
  ``"1,240.00"``, an enum in the wrong case, a scalar returned as a one-element
  list). Repair runs first; validation then judges the repaired record.

Repair never invents a value. If a required field is absent it stays absent and the
record is reported invalid, because a fabricated field is far worse than a missing
one in an extraction system whose whole premise is groundedness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from throughline.schema.spec import Cardinality, ExtractionSchema, FieldSpec, TableSpec


class Severity(str):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Violation:
    """One schema violation, addressed by a dotted path into the record."""

    path: str
    code: str
    message: str
    severity: str = Severity.ERROR

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.message}"


@dataclass
class ValidationReport:
    """The outcome of validating one extraction record."""

    violations: list[Violation] = field(default_factory=list)
    repaired_paths: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """Schema-valid means no errors. Warnings do not invalidate a record."""
        return not self.errors

    def summary(self) -> str:
        if self.is_valid:
            suffix = f" ({len(self.warnings)} warning(s))" if self.warnings else ""
            return f"schema-valid{suffix}"
        return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [{"path": v.path, "code": v.code, "message": v.message} for v in self.errors],
            "warnings": [
                {"path": v.path, "code": v.code, "message": v.message} for v in self.warnings
            ],
            "repaired_paths": list(self.repaired_paths),
        }


def _repair_scalar(spec: FieldSpec, value: Any, path: str, report: ValidationReport) -> Any:
    """Coerce one scalar, recording whether coercion changed it."""
    coerced = spec.coerce(value)
    if coerced is None:
        return None
    if coerced != value:
        report.repaired_paths.append(path)
    return coerced


def _validate_field(
    spec: FieldSpec, record: dict[str, Any], report: ValidationReport
) -> None:
    path = spec.name
    present = spec.name in record and record[spec.name] not in (None, "", [], {})

    if not present:
        if spec.required:
            report.violations.append(
                Violation(path, "missing_required", "required field is absent or empty")
            )
        record.pop(spec.name, None)
        return

    value = record[spec.name]

    if spec.cardinality is Cardinality.MANY:
        # A model that returns a bare scalar for a list field is repairable.
        if not isinstance(value, list):
            value = [value]
            report.repaired_paths.append(path)
        cleaned: list[Any] = []
        for index, item in enumerate(value):
            coerced = _repair_scalar(spec, item, f"{path}[{index}]", report)
            if coerced is None:
                report.violations.append(
                    Violation(
                        f"{path}[{index}]",
                        "type_mismatch",
                        f"value {item!r} is not a valid {spec.type.value}",
                    )
                )
            else:
                cleaned.append(coerced)
        record[spec.name] = cleaned
        if spec.required and not cleaned:
            report.violations.append(
                Violation(path, "missing_required", "required list field has no valid entries")
            )
        return

    # Cardinality ONE / OPTIONAL. A one-element list around a scalar is repairable.
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
            report.repaired_paths.append(path)
        else:
            report.violations.append(
                Violation(path, "cardinality", f"expected a single value, got {len(value)} values")
            )
            return

    coerced = _repair_scalar(spec, value, path, report)
    if coerced is None:
        report.violations.append(
            Violation(path, "type_mismatch", f"value {value!r} is not a valid {spec.type.value}")
        )
        return
    record[spec.name] = coerced


def _validate_table(
    spec: TableSpec, record: dict[str, Any], report: ValidationReport
) -> None:
    path = spec.name
    rows = record.get(spec.name)

    if rows in (None, "", [], {}):
        if spec.required:
            report.violations.append(
                Violation(path, "missing_required", "required table has no rows")
            )
        record.pop(spec.name, None)
        return

    if isinstance(rows, dict):  # a single row returned unwrapped
        rows = [rows]
        report.repaired_paths.append(path)

    if not isinstance(rows, list):
        report.violations.append(Violation(path, "type_mismatch", "table must be a list of rows"))
        return

    known_columns = {column.name for column in spec.columns}
    cleaned_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        row_path = f"{path}[{index}]"
        if not isinstance(row, dict):
            report.violations.append(
                Violation(row_path, "type_mismatch", "table row must be an object")
            )
            continue

        extra = set(row) - known_columns
        for key in sorted(extra):
            report.violations.append(
                Violation(
                    f"{row_path}.{key}",
                    "unknown_column",
                    "column is not declared in the schema and was dropped",
                    Severity.WARNING,
                )
            )

        cleaned: dict[str, Any] = {}
        for column in spec.columns:
            if column.name not in row or row[column.name] in (None, ""):
                continue
            coerced = _repair_scalar(
                column, row[column.name], f"{row_path}.{column.name}", report
            )
            if coerced is None:
                report.violations.append(
                    Violation(
                        f"{row_path}.{column.name}",
                        "type_mismatch",
                        f"value {row[column.name]!r} is not a valid {column.type.value}",
                    )
                )
            else:
                cleaned[column.name] = coerced

        if cleaned:
            cleaned_rows.append(cleaned)

    record[spec.name] = cleaned_rows

    if spec.required and not cleaned_rows:
        report.violations.append(
            Violation(path, "missing_required", "required table has no valid rows")
        )


def validate_record(
    schema: ExtractionSchema, record: dict[str, Any], *, repair: bool = True
) -> tuple[dict[str, Any], ValidationReport]:
    """Validate (and optionally repair) one extraction record against a schema.

    Args:
        schema: The extraction target.
        record: The model-produced record. Not mutated; a copy is returned.
        repair: When false, type coercion is still used to *test* validity but the
            returned record keeps the original values.

    Returns:
        The (possibly repaired) record and its validation report.
    """
    working = dict(record)
    report = ValidationReport()

    declared = set(schema.all_keys)
    for key in sorted(set(working) - declared):
        report.violations.append(
            Violation(key, "unknown_key", "key is not declared in the schema", Severity.WARNING)
        )
        working.pop(key, None)

    for spec in schema.fields:
        _validate_field(spec, working, report)
    for table in schema.tables:
        _validate_table(table, working, report)

    return (working if repair else dict(record)), report


def is_schema_valid(schema: ExtractionSchema, record: dict[str, Any]) -> bool:
    """Convenience predicate used by the early-exit policy and the eval harness."""
    _, report = validate_record(schema, record)
    return report.is_valid
