"""Extraction schema definitions.

A schema is the contract between the caller and the model: it names the fields to
extract, their types, whether they may continue across a page boundary, and how a
value should be validated. Everything downstream - prompt assembly, constrained
decoding, evidence attribution, scoring - is driven from this one object, so that a
new document type is a new schema rather than new code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class FieldType(str, Enum):
    """The value types a scalar field may take."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    CURRENCY = "currency"
    ENUM = "enum"


class Cardinality(str, Enum):
    """How many times a field may appear in one document."""

    ONE = "one"
    """Exactly one value; later page groups refine rather than append."""

    OPTIONAL = "optional"
    """Zero or one value."""

    MANY = "many"
    """A list that page groups append to."""


_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$"),
)

_CURRENCY_RE = re.compile(r"^-?[$€£¥]?\s?[\d,]+(?:\.\d{1,2})?$")


@dataclass(frozen=True)
class FieldSpec:
    """One extractable field.

    Attributes:
        name: Machine name; also the JSON key in the extraction output.
        type: Value type used for coercion and validation.
        description: Natural-language hint shown to the model in the prompt.
        cardinality: Whether the field holds one value or a list.
        required: If true, a missing value makes the record schema-invalid.
        enum_values: Allowed values when ``type`` is ``ENUM``.
        page_hint: Free-text hint about where the field usually appears
            ("first page", "signature block"). Used by relevant-page retrieval.
        keywords: Lexical anchors used by the page retriever to score pages.
        continues_across_pages: True when a value for this field may legitimately
            be assembled from more than one page group.
    """

    name: str
    type: FieldType = FieldType.STRING
    description: str = ""
    cardinality: Cardinality = Cardinality.ONE
    required: bool = False
    enum_values: tuple[str, ...] = ()
    page_hint: str = ""
    keywords: tuple[str, ...] = ()
    continues_across_pages: bool = False

    def __post_init__(self) -> None:
        if self.type is FieldType.ENUM and not self.enum_values:
            raise ValueError(f"Field {self.name!r} is an enum but declares no enum_values.")

    def coerce(self, value: Any) -> Any:
        """Best-effort conversion of a model-produced value to this field's type.

        Returns ``None`` when the value cannot be represented, which the validator
        reports as a type violation rather than silently dropping.
        """
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None

        if self.type is FieldType.STRING:
            return str(value)

        if self.type is FieldType.INTEGER:
            try:
                return int(str(value).replace(",", "").strip())
            except (TypeError, ValueError):
                return None

        if self.type is FieldType.NUMBER:
            try:
                return float(str(value).replace(",", "").strip())
            except (TypeError, ValueError):
                return None

        if self.type is FieldType.BOOLEAN:
            if isinstance(value, bool):
                return value
            lowered = str(value).lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
            return None

        if self.type is FieldType.DATE:
            text = str(value)
            if any(pattern.match(text) for pattern in _DATE_PATTERNS):
                return text
            return None

        if self.type is FieldType.CURRENCY:
            text = str(value)
            return text if _CURRENCY_RE.match(text) else None

        if self.type is FieldType.ENUM:
            text = str(value)
            for allowed in self.enum_values:
                if text.lower() == allowed.lower():
                    return allowed
            return None

        return value

    def json_type(self) -> str:
        """The JSON Schema primitive this field serialises as."""
        return {
            FieldType.INTEGER: "integer",
            FieldType.NUMBER: "number",
            FieldType.BOOLEAN: "boolean",
        }.get(self.type, "string")


@dataclass(frozen=True)
class TableSpec:
    """A repeating row structure that may continue across a page boundary.

    Table continuation is the reason cross-page state exists. A row that starts on
    page 7 and ends on page 8 is one row, and a header repeated at the top of page 8
    is not a new row - ``continuation_markers`` and ``row_key_columns`` are what let
    the merge step tell those cases apart.
    """

    name: str
    columns: tuple[FieldSpec, ...]
    description: str = ""
    required: bool = False
    row_key_columns: tuple[str, ...] = ()
    """Columns that together identify a row, used to deduplicate across groups."""

    continuation_markers: tuple[str, ...] = (
        "continued",
        "continued on next page",
        "cont.",
        "carried forward",
        "brought forward",
    )
    """Phrases that signal this table spills into the following page.

    Deliberately excludes "subtotal": a subtotal line appears both at a page break in
    a long table *and* at the end of a short one, so treating it as a continuation
    signal makes the final totals page look mid-table and pushes the group boundary
    to the wrong place."""

    def column(self, name: str) -> FieldSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError(f"Table {self.name!r} declares no columns.")
        known = {column.name for column in self.columns}
        unknown = set(self.row_key_columns) - known
        if unknown:
            raise ValueError(
                f"Table {self.name!r} row_key_columns reference unknown columns: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class ExtractionSchema:
    """A complete extraction target for one document type."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    fields: tuple[FieldSpec, ...] = ()
    tables: tuple[TableSpec, ...] = ()

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields] + [t.name for t in self.tables]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Schema {self.name!r} has duplicate keys: {sorted(duplicates)}")

    # ── lookup ────────────────────────────────────────────────────────────
    def field(self, name: str) -> FieldSpec | None:
        for spec in self.fields:
            if spec.name == name:
                return spec
        return None

    def table(self, name: str) -> TableSpec | None:
        for spec in self.tables:
            if spec.name == name:
                return spec
        return None

    @property
    def required_keys(self) -> tuple[str, ...]:
        return tuple(
            [f.name for f in self.fields if f.required] + [t.name for t in self.tables if t.required]
        )

    @property
    def all_keys(self) -> tuple[str, ...]:
        return tuple([f.name for f in self.fields] + [t.name for t in self.tables])

    # ── serialisation ─────────────────────────────────────────────────────
    def to_json_schema(self) -> dict[str, Any]:
        """Render as JSON Schema, for constrained decoding backends that accept it."""
        properties: dict[str, Any] = {}
        for spec in self.fields:
            leaf: dict[str, Any] = {"type": spec.json_type()}
            if spec.description:
                leaf["description"] = spec.description
            if spec.type is FieldType.ENUM:
                leaf["enum"] = list(spec.enum_values)
            properties[spec.name] = (
                {"type": "array", "items": leaf} if spec.cardinality is Cardinality.MANY else leaf
            )

        for table in self.tables:
            properties[table.name] = {
                "type": "array",
                "description": table.description,
                "items": {
                    "type": "object",
                    "properties": {
                        column.name: {"type": column.json_type()} for column in table.columns
                    },
                },
            }

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.name,
            "type": "object",
            "properties": properties,
            "required": list(self.required_keys),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "fields": [
                {
                    "name": f.name,
                    "type": f.type.value,
                    "description": f.description,
                    "cardinality": f.cardinality.value,
                    "required": f.required,
                    "enum_values": list(f.enum_values),
                    "page_hint": f.page_hint,
                    "keywords": list(f.keywords),
                    "continues_across_pages": f.continues_across_pages,
                }
                for f in self.fields
            ],
            "tables": [
                {
                    "name": t.name,
                    "description": t.description,
                    "required": t.required,
                    "row_key_columns": list(t.row_key_columns),
                    "continuation_markers": list(t.continuation_markers),
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.type.value,
                            "description": c.description,
                            "enum_values": list(c.enum_values),
                        }
                        for c in t.columns
                    ],
                }
                for t in self.tables
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtractionSchema:
        def build_field(raw: dict[str, Any]) -> FieldSpec:
            return FieldSpec(
                name=raw["name"],
                type=FieldType(raw.get("type", "string")),
                description=raw.get("description", ""),
                cardinality=Cardinality(raw.get("cardinality", "one")),
                required=bool(raw.get("required", False)),
                enum_values=tuple(raw.get("enum_values", ())),
                page_hint=raw.get("page_hint", ""),
                keywords=tuple(raw.get("keywords", ())),
                continues_across_pages=bool(raw.get("continues_across_pages", False)),
            )

        tables = []
        for raw_table in payload.get("tables", []):
            table_kwargs: dict[str, Any] = {
                "name": raw_table["name"],
                "columns": tuple(build_field(c) for c in raw_table["columns"]),
                "description": raw_table.get("description", ""),
                "required": bool(raw_table.get("required", False)),
                "row_key_columns": tuple(raw_table.get("row_key_columns", ())),
            }
            if raw_table.get("continuation_markers"):
                table_kwargs["continuation_markers"] = tuple(raw_table["continuation_markers"])
            tables.append(TableSpec(**table_kwargs))

        return cls(
            name=payload["name"],
            version=payload.get("version", "1.0.0"),
            description=payload.get("description", ""),
            fields=tuple(build_field(f) for f in payload.get("fields", [])),
            tables=tuple(tables),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> ExtractionSchema:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
