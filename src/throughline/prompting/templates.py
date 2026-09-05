"""Prompt assembly.

The prompt for one page group fuses four things:

1. the **schema** - what to extract, with types and continuation rules;
2. the **OCR/layout signal** for the pages in the group, role-tagged and
   block-addressed so the model can cite what it used;
3. the **carry-over state** - what earlier groups already established, and what is
   still missing;
4. the **output contract** - a strict JSON envelope with a parallel evidence map.

Point 2 is what makes this a fusion pipeline rather than a pure-vision one. The
model sees the page image *and* the extracted text blocks. The image carries layout,
stamps, handwriting, and table geometry; the text carries exact characters and, via
block ids, addressable coordinates. Asking the model to cite a block id is what
turns "the model said $41,200" into "the model said $41,200, and here is the line it
read it from".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from throughline.grouping.page_groups import PageGroup
from throughline.schema.spec import Cardinality, ExtractionSchema, FieldSpec, TableSpec
from throughline.state.cross_page import CrossPageState

SYSTEM_PROMPT = """\
You are a document extraction engine. You read a bounded group of pages from a \
longer document and return structured data.

Rules you must follow:
1. Extract only what the provided pages actually say. Never infer, complete, or \
guess a value that is not present. A missing field is correct; an invented one is a \
failure.
2. Cite your evidence. Every value you return must name the block id you read it \
from, using the [block_id|type] markers in the layout text.
3. Respect continuation. If the carry-over notes say a table was still open at the \
page boundary, rows at the top of this group continue that table. A repeated column \
header is not a new row.
4. Do not repeat what is already settled. If the carry-over already has a field and \
these pages do not contradict it, omit the field from your output.
5. Return one JSON object and nothing else. No prose, no markdown fences, no \
explanation before or after."""


def _render_field(spec: FieldSpec) -> str:
    parts = [f"- {spec.name} ({spec.type.value}"]
    if spec.cardinality is Cardinality.MANY:
        parts.append(", list")
    if spec.required:
        parts.append(", required")
    parts.append(")")
    line = "".join(parts)
    if spec.description:
        line += f": {spec.description}"
    if spec.enum_values:
        line += f" One of: {', '.join(spec.enum_values)}."
    if spec.page_hint:
        line += f" Usually found: {spec.page_hint}."
    if spec.continues_across_pages:
        line += " This value may only be final on a later page; report what these pages show."
    return line


def _render_table(spec: TableSpec) -> str:
    lines = [f"- {spec.name} (table{', required' if spec.required else ''}):"]
    if spec.description:
        lines.append(f"    {spec.description}")
    for column in spec.columns:
        column_line = f"    · {column.name} ({column.type.value})"
        if column.description:
            column_line += f" - {column.description}"
        lines.append(column_line)
    if spec.row_key_columns:
        lines.append(f"    Rows are identified by: {', '.join(spec.row_key_columns)}.")
    return "\n".join(lines)


def render_schema(schema: ExtractionSchema) -> str:
    """The schema as instruction text."""
    sections = [f"SCHEMA: {schema.name} v{schema.version}"]
    if schema.description:
        sections.append(schema.description)

    if schema.fields:
        sections.append("\nFields:")
        sections.extend(_render_field(spec) for spec in schema.fields)
    if schema.tables:
        sections.append("\nTables:")
        sections.extend(_render_table(spec) for spec in schema.tables)
    return "\n".join(sections)


def render_output_contract(schema: ExtractionSchema) -> str:
    """The exact JSON envelope the model must produce."""
    example: dict[str, Any] = {"fields": {}, "evidence": {}, "tables": {}, "open_tables": []}

    first_field = schema.fields[0] if schema.fields else None
    if first_field is not None:
        example["fields"][first_field.name] = "<value or omit>"
        example["evidence"][first_field.name] = [
            {"block_id": "p3b7", "quote": "<the exact text you read>", "confidence": 0.0}
        ]
    if schema.tables:
        table = schema.tables[0]
        example["tables"][table.name] = [
            {
                "values": {column.name: "<value>" for column in table.columns[:2]},
                "block_id": "p3b12",
                "confidence": 0.0,
            }
        ]
        example["open_tables"] = [table.name]

    return (
        "OUTPUT FORMAT - return exactly this JSON object shape:\n"
        f"{json.dumps(example, indent=2)}\n\n"
        "- `fields`: only fields you found on THESE pages. Omit anything not present.\n"
        "- `evidence`: for every key in `fields`, the block id(s) you read it from and "
        "the exact quoted text. `confidence` is your own 0.0-1.0 estimate.\n"
        "- `tables`: rows found on these pages, in printed order, each with the block id "
        "of the row.\n"
        "- `open_tables`: name any table that is still mid-flight at the LAST page of "
        "this group, so the next group knows to continue it."
    )


@dataclass
class PromptBundle:
    """An assembled prompt plus the images that go with it."""

    system: str
    user: str
    image_paths: list[str]
    group_index: int
    page_numbers: tuple[int, ...]

    def cache_key_material(self) -> str:
        """What the prompt cache hashes. Images are keyed by path, text by content."""
        return "\x00".join([self.system, self.user, *sorted(self.image_paths)])

    def approximate_tokens(self) -> int:
        """Rough token estimate (~4 chars/token) for budgeting and logs."""
        return (len(self.system) + len(self.user)) // 4 + 300 * len(self.image_paths)


def build_prompt(
    schema: ExtractionSchema,
    group: PageGroup,
    state: CrossPageState | None = None,
    *,
    max_chars_per_page: int = 6_000,
    focus_fields: list[str] | None = None,
) -> PromptBundle:
    """Assemble the prompt for one page group.

    Args:
        schema: The extraction target.
        group: The page group to read.
        state: Accumulated cross-page state; ``None`` for the first group.
        max_chars_per_page: Truncation bound on layout text per page.
        focus_fields: When given, the prompt tells the model to prioritise these -
            used by the orchestrator to steer a group at the fields still missing.

    Returns:
        A :class:`PromptBundle` ready for a VLM backend.
    """
    sections: list[str] = [render_schema(schema), ""]

    if state is not None:
        sections += [
            "CARRY-OVER STATE FROM EARLIER PAGE GROUPS",
            state.render_carry_over(),
            "",
        ]

    if focus_fields:
        sections += [
            "PRIORITY: these keys are still missing. Look for them first: "
            + ", ".join(focus_fields),
            "",
        ]

    span = (
        f"page {group.first_page}"
        if group.first_page == group.last_page
        else f"pages {group.first_page}-{group.last_page}"
    )
    sections += [
        f"CURRENT PAGE GROUP: {span} of the document.",
        "",
        "PAGE TEXT AND LAYOUT (each line is [block_id|role] text):",
        group.render_layout(max_chars_per_page=max_chars_per_page),
        "",
        render_output_contract(schema),
    ]

    return PromptBundle(
        system=SYSTEM_PROMPT,
        user="\n".join(sections),
        image_paths=group.image_paths(),
        group_index=group.group_index,
        page_numbers=group.page_numbers,
    )


def build_repair_prompt(
    schema: ExtractionSchema, raw_output: str, errors: list[str]
) -> PromptBundle:
    """A short second-pass prompt asking the model to fix a schema-invalid output.

    Cheap compared with re-reading the pages: it carries no images and no layout
    text, only the previous output and what was wrong with it.
    """
    user = "\n".join(
        [
            "Your previous output did not satisfy the schema.",
            "",
            "Errors:",
            *(f"  - {message}" for message in errors),
            "",
            "Your previous output:",
            raw_output.strip()[:4_000],
            "",
            render_schema(schema),
            "",
            "Return the corrected JSON object only. Do not add fields you did not "
            "originally extract - fix the shape, not the content.",
        ]
    )
    return PromptBundle(
        system=SYSTEM_PROMPT, user=user, image_paths=[], group_index=-1, page_numbers=()
    )
