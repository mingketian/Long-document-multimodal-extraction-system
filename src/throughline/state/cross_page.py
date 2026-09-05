"""Cross-page extraction state.

This is the object that makes bounded processing possible. Each page group is read
in isolation, but the state carries forward what has already been found, so that:

* a later group **refines** a field rather than re-extracting it from nothing;
* a table that spans groups **accumulates** rows instead of restarting;
* every value keeps a **provenance trail** - which group produced it, from which
  page, quoting which block - so the final answer is attributable;
* the prompt for group *n+1* can be told what is already known and, crucially, what
  is still missing, which is what turns a sequence of independent reads into one
  coherent extraction.

The state is deliberately small when rendered. A carry-over that grows with the
document would defeat the bound that page grouping exists to enforce, so
:meth:`CrossPageState.render_carry_over` summarises rather than dumps.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from throughline.schema.spec import Cardinality, ExtractionSchema, TableSpec


@dataclass(frozen=True)
class EvidenceRef:
    """A pointer from an extracted value back to the text that supports it."""

    page_number: int
    block_id: str | None = None
    quote: str = ""
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 1.0

    def cite(self) -> str:
        """Short human-readable citation, e.g. ``p12:b7``."""
        return f"p{self.page_number}" + (f":{self.block_id}" if self.block_id else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "block_id": self.block_id,
            "quote": self.quote,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceRef:
        bbox = payload.get("bbox")
        return cls(
            page_number=int(payload["page_number"]),
            block_id=payload.get("block_id"),
            quote=payload.get("quote", ""),
            bbox=tuple(bbox) if bbox else None,  # type: ignore[arg-type]
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass
class FieldValue:
    """One field's current value, with everything needed to defend or replace it."""

    name: str
    value: Any
    confidence: float = 0.0
    source_group: int = -1
    evidence: list[EvidenceRef] = field(default_factory=list)
    revision_count: int = 0
    """How many times a later group overwrote this value. High counts mean the field
    is genuinely ambiguous in the document and are worth surfacing."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "source_group": self.source_group,
            "revision_count": self.revision_count,
            "evidence": [ref.to_dict() for ref in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FieldValue:
        return cls(
            name=payload["name"],
            value=payload.get("value"),
            confidence=float(payload.get("confidence", 0.0)),
            source_group=int(payload.get("source_group", -1)),
            revision_count=int(payload.get("revision_count", 0)),
            evidence=[EvidenceRef.from_dict(r) for r in payload.get("evidence", [])],
        )


@dataclass
class TableRow:
    """One accumulated table row plus the provenance of where it came from."""

    values: dict[str, Any]
    source_group: int = -1
    page_number: int = 0
    evidence: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0

    def key(self, table: TableSpec) -> tuple[Any, ...]:
        """Identity of this row, used to detect a repeat across an overlap.

        Falls back to the whole row when the schema declares no key columns, which
        makes deduplication exact-match only - correct, just less forgiving.
        """
        columns = table.row_key_columns or tuple(c.name for c in table.columns)
        return tuple(_normalise_key(self.values.get(column)) for column in columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "source_group": self.source_group,
            "page_number": self.page_number,
            "confidence": self.confidence,
            "evidence": [ref.to_dict() for ref in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TableRow:
        return cls(
            values=dict(payload.get("values", {})),
            source_group=int(payload.get("source_group", -1)),
            page_number=int(payload.get("page_number", 0)),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=[EvidenceRef.from_dict(r) for r in payload.get("evidence", [])],
        )


def _normalise_key(value: Any) -> Any:
    """Loose normalisation so ``"Widget A"`` and ``"widget  a"`` match as one row."""
    if value is None:
        return None
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass
class CrossPageState:
    """Accumulated extraction state for one document."""

    schema: ExtractionSchema
    document_id: str = ""
    fields: dict[str, FieldValue] = field(default_factory=dict)
    tables: dict[str, list[TableRow]] = field(default_factory=dict)
    open_tables: set[str] = field(default_factory=set)
    """Tables the previous group left mid-flight; the next prompt is told to continue
    them rather than treat a repeated header as a fresh table."""

    groups_processed: list[int] = field(default_factory=list)
    pages_seen: set[int] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    # ── ingestion ─────────────────────────────────────────────────────────
    def update_field(
        self,
        name: str,
        value: Any,
        *,
        group_index: int,
        confidence: float = 0.0,
        evidence: Iterable[EvidenceRef] = (),
    ) -> bool:
        """Merge one field observation into the state.

        Returns True when the state actually changed. The resolution rule depends on
        cardinality: ``MANY`` fields append (deduplicated), scalar fields keep the
        higher-confidence value and only fall back to recency on a tie.
        """
        spec = self.schema.field(name)
        if spec is None or value in (None, "", [], {}):
            return False

        evidence = list(evidence)

        if spec.cardinality is Cardinality.MANY:
            return self._append_field(name, value, group_index, confidence, evidence)

        existing = self.fields.get(name)
        if existing is None:
            self.fields[name] = FieldValue(
                name=name,
                value=value,
                confidence=confidence,
                source_group=group_index,
                evidence=evidence,
            )
            return True

        if _normalise_key(existing.value) == _normalise_key(value):
            # Same answer from a second group: corroboration, not a revision.
            existing.confidence = max(existing.confidence, confidence)
            existing.evidence.extend(ref for ref in evidence if ref not in existing.evidence)
            return False

        # A field the schema marks as continuing across pages prefers the later
        # reading: totals and closing balances are only correct once the last
        # continuation page has been seen.
        prefer_new = (
            confidence > existing.confidence
            or (spec.continues_across_pages and confidence >= existing.confidence)
        )
        if not prefer_new:
            return False

        self.notes.append(
            f"field {name!r} revised in group {group_index}: "
            f"{existing.value!r} -> {value!r} (conf {existing.confidence:.2f} -> {confidence:.2f})"
        )
        existing.value = value
        existing.confidence = confidence
        existing.source_group = group_index
        existing.evidence = evidence or existing.evidence
        existing.revision_count += 1
        return True

    def _append_field(
        self,
        name: str,
        value: Any,
        group_index: int,
        confidence: float,
        evidence: list[EvidenceRef],
    ) -> bool:
        entry = self.fields.get(name)
        if entry is None:
            entry = FieldValue(
                name=name, value=[], confidence=confidence, source_group=group_index
            )
            self.fields[name] = entry
        if not isinstance(entry.value, list):
            entry.value = [entry.value]

        incoming = value if isinstance(value, list) else [value]
        seen = {_normalise_key(item) for item in entry.value}
        changed = False
        for item in incoming:
            if _normalise_key(item) in seen:
                continue
            entry.value.append(item)
            seen.add(_normalise_key(item))
            changed = True
        if changed:
            entry.confidence = max(entry.confidence, confidence)
            entry.source_group = group_index
            entry.evidence.extend(evidence)
        return changed

    def append_rows(
        self,
        table_name: str,
        rows: Iterable[dict[str, Any]],
        *,
        group_index: int,
        page_number: int = 0,
        evidence: Iterable[EvidenceRef] = (),
        confidence: float = 0.0,
    ) -> int:
        """Append table rows, dropping repeats produced by the page overlap.

        Returns the number of genuinely new rows added.
        """
        table = self.schema.table(table_name)
        if table is None:
            return 0

        bucket = self.tables.setdefault(table_name, [])
        existing_keys = {row.key(table) for row in bucket}
        evidence = list(evidence)
        added = 0

        for raw in rows:
            if not isinstance(raw, dict) or not any(
                value not in (None, "") for value in raw.values()
            ):
                continue
            candidate = TableRow(
                values=raw,
                source_group=group_index,
                page_number=page_number,
                evidence=evidence,
                confidence=confidence,
            )
            key = candidate.key(table)
            if key in existing_keys and any(part is not None for part in key):
                continue
            bucket.append(candidate)
            existing_keys.add(key)
            added += 1

        return added

    def mark_table_open(self, table_name: str, *, open_: bool = True) -> None:
        """Record whether a table is still mid-flight at the group boundary."""
        if open_:
            self.open_tables.add(table_name)
        else:
            self.open_tables.discard(table_name)

    def record_group(self, group_index: int, page_numbers: Iterable[int]) -> None:
        if group_index not in self.groups_processed:
            self.groups_processed.append(group_index)
        self.pages_seen.update(page_numbers)

    # ── inspection ────────────────────────────────────────────────────────
    def missing_required(self) -> list[str]:
        """Required keys with nothing in them yet - the early-exit gate reads this."""
        missing = []
        for name in self.schema.required_keys:
            if name in self.fields and self.fields[name].value not in (None, "", [], {}):
                continue
            if self.tables.get(name):
                continue
            missing.append(name)
        return missing

    def missing_optional(self) -> list[str]:
        filled = set(self.fields) | {name for name, rows in self.tables.items() if rows}
        return [key for key in self.schema.all_keys if key not in filled]

    def coverage(self) -> float:
        """Fraction of declared keys that currently hold a value."""
        keys = self.schema.all_keys
        if not keys:
            return 1.0
        filled = sum(
            1
            for key in keys
            if (key in self.fields and self.fields[key].value not in (None, "", [], {}))
            or self.tables.get(key)
        )
        return filled / len(keys)

    def mean_confidence(self) -> float:
        scores = [entry.confidence for entry in self.fields.values()]
        scores += [row.confidence for rows in self.tables.values() for row in rows]
        return sum(scores) / len(scores) if scores else 0.0

    def row_count(self, table_name: str) -> int:
        return len(self.tables.get(table_name, []))

    # ── outputs ───────────────────────────────────────────────────────────
    def to_record(self) -> dict[str, Any]:
        """The extraction result, in schema shape and nothing else."""
        record: dict[str, Any] = {}
        for name, entry in self.fields.items():
            if entry.value not in (None, "", [], {}):
                record[name] = entry.value
        for name, rows in self.tables.items():
            if rows:
                record[name] = [row.values for row in rows]
        return record

    def evidence_map(self) -> dict[str, list[dict[str, Any]]]:
        """Every extracted key mapped to the evidence supporting it."""
        mapping: dict[str, list[dict[str, Any]]] = {}
        for name, entry in self.fields.items():
            if entry.evidence:
                mapping[name] = [ref.to_dict() for ref in entry.evidence]
        for name, rows in self.tables.items():
            refs = [ref.to_dict() for row in rows for ref in row.evidence]
            if refs:
                mapping[name] = refs
        return mapping

    def render_carry_over(self, *, max_chars: int = 1_800, max_rows_shown: int = 3) -> str:
        """Compact summary of known state, injected into the next group's prompt.

        Deliberately lossy. The next group needs to know *what is settled*, *what is
        still open*, and *where a table left off* - not the full record. Keeping this
        bounded is what stops the prompt growing with the document.
        """
        if not self.fields and not self.tables:
            return "(nothing extracted yet - this is the first page group)"

        lines: list[str] = []

        if self.fields:
            lines.append("Already extracted (do not re-derive unless you find better evidence):")
            for name in sorted(self.fields):
                entry = self.fields[name]
                if entry.value in (None, "", [], {}):
                    continue
                cites = ", ".join(ref.cite() for ref in entry.evidence[:2])
                shown = _abbreviate(entry.value)
                suffix = f"  [{cites}]" if cites else ""
                lines.append(f"  - {name}: {shown}{suffix}")

        for name, rows in self.tables.items():
            if not rows:
                continue
            lines.append(f"Table {name!r}: {len(rows)} row(s) captured so far.")
            table = self.schema.table(name)
            key_columns = (table.row_key_columns if table else ()) or ()
            for row in rows[-max_rows_shown:]:
                if key_columns:
                    preview = ", ".join(
                        f"{column}={row.values.get(column)!r}" for column in key_columns
                    )
                else:
                    preview = _abbreviate(row.values)
                lines.append(f"    last: {preview} (page {row.page_number})")

        if self.open_tables:
            lines.append(
                "CONTINUING: "
                + ", ".join(sorted(self.open_tables))
                + " ran past the previous page boundary. Rows at the top of this group "
                "continue that table; a repeated column header is not a new row."
            )

        missing = self.missing_required()
        if missing:
            lines.append("Still missing (required): " + ", ".join(missing))

        rendered = "\n".join(lines)
        if len(rendered) > max_chars:
            rendered = rendered[: max_chars - 20].rstrip() + "\n  [...truncated]"
        return rendered

    # ── serialisation ─────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "schema": self.schema.name,
            "schema_version": self.schema.version,
            "fields": {name: entry.to_dict() for name, entry in self.fields.items()},
            "tables": {
                name: [row.to_dict() for row in rows] for name, rows in self.tables.items()
            },
            "open_tables": sorted(self.open_tables),
            "groups_processed": list(self.groups_processed),
            "pages_seen": sorted(self.pages_seen),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], schema: ExtractionSchema) -> CrossPageState:
        state = cls(schema=schema, document_id=payload.get("document_id", ""))
        state.fields = {
            name: FieldValue.from_dict(raw) for name, raw in payload.get("fields", {}).items()
        }
        state.tables = {
            name: [TableRow.from_dict(raw) for raw in rows]
            for name, rows in payload.get("tables", {}).items()
        }
        state.open_tables = set(payload.get("open_tables", []))
        state.groups_processed = list(payload.get("groups_processed", []))
        state.pages_seen = set(payload.get("pages_seen", []))
        state.notes = list(payload.get("notes", []))
        return state

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def _abbreviate(value: Any, *, limit: int = 90) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."
