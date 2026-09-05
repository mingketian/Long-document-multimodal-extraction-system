"""A deterministic, dependency-free backend.

This is **not** a language model and does not pretend to be one. It is a rule-based
extractor that reads the same prompt a VLM would receive, pulls values out of the
layout text using each field's declared keywords, and emits the same JSON envelope
with real block-id citations.

It exists for three reasons, all of them practical:

* **The pipeline is runnable by anyone.** ``throughline extract`` works after a
  plain ``pip install`` - no GPU, no weights, no AWS account.
* **CI can test the whole path.** Grouping, cross-page state, table continuation,
  validation, attribution and early exit are all exercised end to end, deterministically.
* **It is an honest floor.** When a fine-tuned checkpoint is evaluated, this is the
  no-model baseline the gain is measured against. Keyword matching does recover a
  surprising number of header fields; what it cannot do is read a table that
  continues across a page break, or ground a value it did not lexically match -
  which is precisely the gap the VLM is there to close.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from throughline.models.base import GenerationConfig, GenerationResult, timed
from throughline.prompting.templates import PromptBundle
from throughline.schema.spec import Cardinality, ExtractionSchema, FieldSpec, FieldType

_BLOCK_LINE = re.compile(r"^\[(?P<block_id>[^|\]]+)\|(?P<role>[^\]]+)\]\s*(?P<text>.*)$")
_PAGE_LINE = re.compile(r"^---\s*PAGE\s+(?P<page>\d+)")

_VALUE_PATTERNS: dict[FieldType, re.Pattern[str]] = {
    FieldType.DATE: re.compile(
        r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
    ),
    FieldType.CURRENCY: re.compile(r"(-?[$€£¥]?\s?[\d,]+\.\d{2}|-?[$€£¥]\s?[\d,]+)"),
    FieldType.INTEGER: re.compile(r"(-?\d[\d,]*)"),
    FieldType.NUMBER: re.compile(r"(-?\d[\d,]*(?:\.\d+)?)"),
}

# A currency value carrying its symbol is a far better match than a bare number,
# which in practice is usually a tax *rate* sitting next to the tax *amount*.
_SYMBOLED_CURRENCY = re.compile(r"(-?[$€£¥]\s?[\d,]+(?:\.\d{2})?)")


def _keyword_position(text: str, keyword: str) -> int:
    """Find a keyword on a word boundary, so "total" does not match "Subtotal"."""
    match = re.search(rf"(?<![A-Za-z]){re.escape(keyword)}(?![A-Za-z])", text, re.IGNORECASE)
    return match.start() if match else -1


@dataclass(frozen=True)
class _Line:
    """One parsed layout line from the prompt."""

    page: int
    block_id: str
    role: str
    text: str


def _parse_prompt_layout(user_prompt: str) -> list[_Line]:
    """Recover the block-addressed layout lines the prompt builder rendered."""
    lines: list[_Line] = []
    page = 0
    for raw in user_prompt.splitlines():
        page_match = _PAGE_LINE.match(raw.strip())
        if page_match:
            page = int(page_match.group("page"))
            continue
        block_match = _BLOCK_LINE.match(raw.strip())
        if block_match and page:
            lines.append(
                _Line(
                    page=page,
                    block_id=block_match.group("block_id"),
                    role=block_match.group("role"),
                    text=block_match.group("text").strip(),
                )
            )
    return lines


def _value_after_keyword(text: str, keyword: str, spec: FieldSpec) -> str | None:
    """Pull the value that follows a keyword on the same line.

    Handles the two shapes that cover most business documents: ``Label: value`` and
    ``Label   value`` (column-aligned).
    """
    position = _keyword_position(text, keyword)
    if position < 0:
        return None

    tail = text[position + len(keyword) :].lstrip(" :\t#.-")
    if not tail:
        return None

    if spec.type is FieldType.CURRENCY:
        symboled = _SYMBOLED_CURRENCY.search(tail)
        if symboled:
            return symboled.group(1).strip()

    pattern = _VALUE_PATTERNS.get(spec.type)
    if pattern is not None:
        match = pattern.search(tail)
        return match.group(1).strip() if match else None

    if spec.type is FieldType.ENUM:
        for allowed in spec.enum_values:
            if re.search(rf"\b{re.escape(allowed)}\b", tail, re.IGNORECASE):
                return allowed
        return None

    if spec.type is FieldType.BOOLEAN:
        lowered_tail = tail.lower()
        if any(word in lowered_tail for word in ("yes", "true", "shall automatically")):
            return "true"
        if any(word in lowered_tail for word in ("no", "false", "shall not")):
            return "false"
        return None

    # Free text: take the remainder of the line, bounded.
    return tail.split("  ")[0].strip()[:200] or None


@dataclass
class RuleBasedBackend:
    """Keyword-and-regex extraction over the prompt's layout text."""

    name: str = "rule-based"
    confidence: float = 0.55
    """Reported per-value confidence. Deliberately mid-range: these values are real
    matches, but a lexical hit is much weaker evidence than a model reading a page."""

    max_table_rows_per_group: int = 200
    _calls: int = field(default=0, init=False)

    @timed
    def generate(
        self, prompt: PromptBundle, config: GenerationConfig | None = None
    ) -> GenerationResult:
        self._calls += 1
        schema = _schema_from_prompt(prompt)
        lines = _parse_prompt_layout(prompt.user)

        fields: dict[str, Any] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        tables: dict[str, list[dict[str, Any]]] = {}
        open_tables: list[str] = []

        if schema is not None:
            fields, evidence = self._extract_fields(schema, lines)
            tables, open_tables = self._extract_tables(schema, lines)

        payload = {
            "fields": fields,
            "evidence": evidence,
            "tables": tables,
            "open_tables": open_tables,
        }
        text = json.dumps(payload, indent=2)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt.approximate_tokens(),
            completion_tokens=len(text) // 4,
            backend=self.name,
            metadata={"call_index": self._calls, "lines_parsed": len(lines)},
        )

    # ── fields ────────────────────────────────────────────────────────────
    def _extract_fields(
        self, schema: ExtractionSchema, lines: list[_Line]
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        fields: dict[str, Any] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}

        for spec in schema.fields:
            anchors = list(spec.keywords) or [spec.name.replace("_", " ")]
            found: list[tuple[str, _Line]] = []

            for line in lines:
                for anchor in anchors:
                    value = _value_after_keyword(line.text, anchor, spec)
                    if value:
                        found.append((value, line))
                        break
                if found and spec.cardinality is not Cardinality.MANY:
                    break

            if not found:
                continue

            if spec.cardinality is Cardinality.MANY:
                values = list(dict.fromkeys(value for value, _ in found))
                fields[spec.name] = values
                evidence[spec.name] = [
                    {
                        "block_id": line.block_id,
                        "quote": line.text[:200],
                        "confidence": self.confidence,
                    }
                    for _, line in found[: len(values)]
                ]
            else:
                value, line = found[0]
                fields[spec.name] = value
                evidence[spec.name] = [
                    {
                        "block_id": line.block_id,
                        "quote": line.text[:200],
                        "confidence": self.confidence,
                    }
                ]

        return fields, evidence

    # ── tables ────────────────────────────────────────────────────────────
    def _extract_tables(
        self, schema: ExtractionSchema, lines: list[_Line]
    ) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
        tables: dict[str, list[dict[str, Any]]] = {}
        open_tables: list[str] = []
        if not schema.tables or not lines:
            return tables, open_tables

        table = schema.tables[0]
        rows: list[dict[str, Any]] = []
        row_lines = [line for line in lines if line.role in {"table_row", "table"}]

        for line in row_lines[: self.max_table_rows_per_group]:
            values = _split_row(line.text, table.columns)
            if not values:
                continue
            rows.append(
                {
                    "values": values,
                    "block_id": line.block_id,
                    "page": line.page,
                    "confidence": self.confidence,
                }
            )

        if rows:
            tables[table.name] = rows
            last_page = max(line.page for line in lines)
            tail = " ".join(line.text.lower() for line in lines if line.page == last_page)
            if any(marker in tail for marker in table.continuation_markers) or row_lines and row_lines[-1].page == last_page:
                open_tables.append(table.name)

        return tables, open_tables


def _split_row(text: str, columns: tuple[FieldSpec, ...]) -> dict[str, Any]:
    """Split a printed table row on runs of whitespace or pipes."""
    cells = [cell.strip() for cell in re.split(r"\s{2,}|\t|\s*\|\s*", text) if cell.strip()]
    if len(cells) < 2:
        return {}

    values: dict[str, Any] = {}
    for column, cell in zip(columns, cells, strict=False):  # a short row is fine
        pattern = _VALUE_PATTERNS.get(column.type)
        if pattern is not None:
            match = pattern.search(cell)
            if match:
                values[column.name] = match.group(1)
            continue
        values[column.name] = cell

    # A row that yielded only a header word is a repeated column header, not data.
    header_words = {column.name.replace("_", " ").lower() for column in columns}
    if all(str(value).lower() in header_words for value in values.values()):
        return {}
    return values


_SCHEMA_LINE = re.compile(r"^SCHEMA:\s*(?P<name>\S+)\s+v(?P<version>\S+)")


def _schema_from_prompt(prompt: PromptBundle) -> ExtractionSchema | None:
    """Recover the schema the prompt was built from, via the registry."""
    from throughline.schema import registry

    for line in prompt.user.splitlines():
        match = _SCHEMA_LINE.match(line.strip())
        if match:
            try:
                return registry.get(match.group("name"))
            except KeyError:
                return None
    return None


@dataclass
class EchoBackend:
    """Returns a fixed response. Used to test error handling and repair paths."""

    response: str = "{}"
    name: str = "echo"

    @timed
    def generate(
        self, prompt: PromptBundle, config: GenerationConfig | None = None
    ) -> GenerationResult:
        return GenerationResult(text=self.response, backend=self.name)
