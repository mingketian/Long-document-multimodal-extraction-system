"""Schema-constrained output handling.

"Schema-constrained extraction" means two different things depending on where you
stand in the stack, and this module implements both:

* **Hard constraint at decode time.** When the serving stack supports it, the schema
  is compiled to a grammar and the sampler is prevented from emitting a token that
  would break it. :func:`build_grammar` produces the JSON Schema that vLLM,
  SGLang, Outlines and TGI all accept for this.

* **Soft constraint at parse time.** When it does not - and a hosted endpoint often
  does not - the output is parsed defensively and repaired. That is what
  :func:`parse_envelope` does: strip markdown fences, find the outermost JSON object,
  fix the specific malformations models actually produce (trailing commas, single
  quotes, unquoted keys, a truncated tail), and only then give up.

The measured effect of the two together is a schema-valid rate that stops depending
on whether the model happened to close its braces.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from throughline.schema.spec import ExtractionSchema

LOGGER = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*(?=[}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")


class ParseError(ValueError):
    """Raised when a model output cannot be recovered as JSON."""


@dataclass
class ParsedEnvelope:
    """The normalised result of one model call."""

    fields: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    open_tables: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def was_repaired(self) -> bool:
        return bool(self.repairs)

    def is_empty(self) -> bool:
        return not self.fields and not self.tables


# ── hard constraint ───────────────────────────────────────────────────────
def build_grammar(schema: ExtractionSchema) -> dict[str, Any]:
    """JSON Schema for the full output envelope, for grammar-constrained decoding.

    Pass to vLLM as ``guided_json``, to SGLang as ``json_schema``, or to Outlines as
    the schema argument. The envelope - not just the record - is constrained, so the
    evidence map is as guaranteed as the values it supports.
    """
    record = schema.to_json_schema()
    evidence_item = {
        "type": "object",
        "properties": {
            "block_id": {"type": "string"},
            "quote": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["block_id"],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{schema.name}_group_envelope",
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": record["properties"],
                "additionalProperties": False,
            },
            "evidence": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": evidence_item},
            },
            "tables": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "values": {"type": "object"},
                            "block_id": {"type": "string"},
                            "page": {"type": "integer"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["values"],
                    },
                },
            },
            "open_tables": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


# ── soft constraint ───────────────────────────────────────────────────────
def _strip_fences(text: str) -> tuple[str, bool]:
    match = _FENCE_RE.search(text)
    if match:
        return match.group("body").strip(), True
    return text.strip(), False


def _outermost_object(text: str) -> tuple[str, bool]:
    """Find the outermost balanced ``{...}``, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        raise ParseError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1 < len(text.rstrip())

    # Unbalanced: the generation was cut off. Close what is open and salvage it.
    salvage = text[start:].rstrip().rstrip(",")
    if in_string:
        salvage += '"'
    salvage += "}" * depth
    return salvage, True


def _repair_json(text: str, repairs: list[str]) -> Any:
    """Try increasingly aggressive fixes, recording each one that was needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate = _TRAILING_COMMA_RE.sub("", text)
    if candidate != text:
        repairs.append("removed trailing comma")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    quoted = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', candidate)
    if quoted != candidate:
        repairs.append("quoted bare keys")
        try:
            return json.loads(quoted)
        except json.JSONDecodeError:
            pass
        candidate = quoted

    if "'" in candidate and '"' not in candidate:
        single = candidate.replace("'", '"')
        repairs.append("converted single quotes")
        try:
            return json.loads(single)
        except json.JSONDecodeError:
            pass

    normalised = re.sub(r"\b(NaN|Infinity|-Infinity)\b", "null", candidate)
    normalised = re.sub(r"\b(True|False|None)\b", lambda m: {
        "True": "true", "False": "false", "None": "null"
    }[m.group(1)], normalised)
    if normalised != candidate:
        repairs.append("normalised python literals")
        try:
            return json.loads(normalised)
        except json.JSONDecodeError:
            pass

    raise ParseError(f"Could not parse model output as JSON: {text[:300]}")


def _coerce_evidence(raw: Any) -> dict[str, list[dict[str, Any]]]:
    """Accept the several evidence shapes models produce; normalise to one."""
    evidence: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        return evidence

    for key, value in raw.items():
        items: list[dict[str, Any]] = []
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, dict):
                items.append(candidate)
            elif isinstance(candidate, str):
                # A bare block id, or a bare quote.
                if re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", candidate):
                    items.append({"block_id": candidate})
                else:
                    items.append({"quote": candidate})
        if items:
            evidence[str(key)] = items
    return evidence


def _coerce_tables(raw: Any) -> dict[str, list[dict[str, Any]]]:
    """Accept either ``[{values: {...}}]`` or a bare list of row objects."""
    tables: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        return tables

    for name, rows in raw.items():
        if not isinstance(rows, list):
            continue
        normalised: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if "values" in row and isinstance(row["values"], dict):
                normalised.append(row)
            else:
                normalised.append({"values": row})
        if normalised:
            tables[str(name)] = normalised
    return tables


def parse_envelope(text: str, schema: ExtractionSchema | None = None) -> ParsedEnvelope:
    """Parse one model output into a :class:`ParsedEnvelope`.

    Args:
        text: Raw backend output.
        schema: When given, keys not declared in the schema are dropped rather than
            carried into state.

    Raises:
        ParseError: When no JSON object can be recovered at all.
    """
    repairs: list[str] = []
    stripped, fenced = _strip_fences(text)
    if fenced:
        repairs.append("stripped markdown fence")

    body, had_trailing = _outermost_object(stripped)
    if had_trailing:
        repairs.append("extracted outermost object")

    payload = _repair_json(body, repairs)
    if not isinstance(payload, dict):
        raise ParseError(f"Model output parsed to {type(payload).__name__}, not an object.")

    # A model that returns the record directly instead of the envelope is common
    # enough to be worth handling rather than failing.
    if "fields" not in payload and schema is not None:
        declared = set(schema.all_keys)
        if declared & set(payload):
            repairs.append("wrapped bare record in envelope")
            table_names = {table.name for table in schema.tables}
            payload = {
                "fields": {k: v for k, v in payload.items() if k not in table_names},
                "tables": {
                    k: [{"values": row} for row in v]
                    for k, v in payload.items()
                    if k in table_names and isinstance(v, list)
                },
            }

    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    tables = _coerce_tables(payload.get("tables"))
    evidence = _coerce_evidence(payload.get("evidence"))
    open_tables = payload.get("open_tables")
    open_tables = [str(name) for name in open_tables] if isinstance(open_tables, list) else []

    if schema is not None:
        declared = set(schema.all_keys)
        dropped = set(fields) - declared
        if dropped:
            repairs.append(f"dropped undeclared keys: {sorted(dropped)}")
            fields = {k: v for k, v in fields.items() if k in declared}
        tables = {k: v for k, v in tables.items() if k in declared}
        evidence = {k: v for k, v in evidence.items() if k in declared}

    return ParsedEnvelope(
        fields=dict(fields),
        evidence=evidence,
        tables=tables,
        open_tables=open_tables,
        repairs=repairs,
        raw=text,
    )
