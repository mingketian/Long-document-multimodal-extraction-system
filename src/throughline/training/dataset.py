"""Training-set construction.

The unit of training is a **page-group set**: one bounded window of pages, the
carry-over state a real run would have had at that point, and the structured target
for exactly that window. Training on whole documents would teach the model nothing
about continuation, because it would never see a boundary; training on single pages
would teach it that tables end at page breaks, which is the error we most need it
not to make.

So each example is built by replaying the real pipeline's grouping over a labelled
document and, for every group, asking: given what was known before this window, what
should the model output for this window? That makes training data and inference data
the same shape by construction - the prompt the model learns on is the prompt it will
see.

Two details do most of the work:

* **Carry-over is simulated, not idealised.** The state fed to group *n* holds only
  what groups *0..n-1* could actually have produced, so the model learns to work from
  partial information rather than from a complete answer it will never have.
* **Continuation is labelled explicitly.** ``open_tables`` in the target teaches the
  model to *announce* that a table is still running - the signal the orchestrator's
  early-exit policy depends on.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from throughline.grouping.page_groups import GroupingConfig, PageGroup, partition
from throughline.ingest.layout import Document
from throughline.prompting.templates import build_prompt
from throughline.schema.spec import Cardinality, ExtractionSchema
from throughline.state.cross_page import CrossPageState, EvidenceRef

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """One page-group set, ready for supervised fine-tuning."""

    document_id: str
    group_index: int
    page_numbers: tuple[int, ...]
    system: str
    user: str
    target: str
    image_paths: list[str] = field(default_factory=list)

    def to_messages(self) -> list[dict[str, Any]]:
        """Chat format consumed by the SFT trainer."""
        content: list[dict[str, Any]] = [
            {"type": "image", "image": path} for path in self.image_paths
        ]
        content.append({"type": "text", "text": self.user})
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": self.target}]},
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "group_index": self.group_index,
            "page_numbers": list(self.page_numbers),
            "messages": self.to_messages(),
            "images": list(self.image_paths),
        }

    def approximate_tokens(self) -> int:
        return (len(self.system) + len(self.user) + len(self.target)) // 4


def _pages_of(evidence: Any) -> set[int]:
    pages: set[int] = set()
    items = evidence if isinstance(evidence, list) else [evidence]
    for item in items:
        if isinstance(item, dict) and item.get("page_number") is not None:
            pages.add(int(item["page_number"]))
        elif isinstance(item, int):
            pages.add(item)
    return pages


def _evidence_claims_for(
    key: str, gold_evidence: dict[str, Any], group: PageGroup
) -> list[dict[str, Any]]:
    """Gold citations that fall inside this page group."""
    claims: list[dict[str, Any]] = []
    entries = gold_evidence.get(key)
    if entries is None:
        return claims
    for entry in entries if isinstance(entries, list) else [entries]:
        if not isinstance(entry, dict):
            continue
        page = entry.get("page_number")
        if page is None or int(page) not in group.page_numbers:
            continue
        claims.append(
            {
                "block_id": entry.get("block_id"),
                "quote": entry.get("quote", ""),
                "confidence": 1.0,
            }
        )
    return claims


def _rows_in_group(rows: Sequence[dict[str, Any]], row_pages: Sequence[int], group: PageGroup) -> list[int]:
    """Indices of gold table rows that belong to this group's pages."""
    wanted = set(group.page_numbers)
    if not row_pages:
        return list(range(len(rows)))
    return [index for index, page in enumerate(row_pages) if int(page) in wanted]


def build_target(
    schema: ExtractionSchema,
    group: PageGroup,
    gold: dict[str, Any],
    gold_evidence: dict[str, Any],
    already_extracted: set[str],
    *,
    row_pages: dict[str, Sequence[int]] | None = None,
) -> tuple[str, set[str], list[str]]:
    """Build the assistant target for one page group.

    Returns the serialised target, the keys it newly settles, and the tables it
    leaves open. A key is included only when this group's pages actually carry its
    gold evidence *and* an earlier group has not already settled it - which is what
    teaches the model rule 4 of the system prompt ("do not repeat what is settled")
    instead of merely telling it.
    """
    row_pages = row_pages or {}
    fields: dict[str, Any] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    settled: set[str] = set()

    for spec in schema.fields:
        if spec.name in already_extracted and not spec.continues_across_pages:
            continue
        value = gold.get(spec.name)
        if value in (None, "", [], {}):
            continue

        claims = _evidence_claims_for(spec.name, gold_evidence, group)
        if gold_evidence and not claims:
            # Evidence exists but not on these pages: this group must not claim it.
            continue
        if not gold_evidence and group.group_index != 0:
            # No evidence labels at all: attribute scalars to the first group only.
            continue

        fields[spec.name] = value
        if claims:
            evidence[spec.name] = claims
        settled.add(spec.name)

    tables: dict[str, list[dict[str, Any]]] = {}
    open_tables: list[str] = []

    for table in schema.tables:
        gold_rows = gold.get(table.name)
        if not isinstance(gold_rows, list) or not gold_rows:
            continue

        indices = _rows_in_group(gold_rows, row_pages.get(table.name, ()), group)
        if not indices:
            continue

        rows_payload = []
        for index in indices:
            row = gold_rows[index]
            if not isinstance(row, dict):
                continue
            pages = row_pages.get(table.name, ())
            rows_payload.append(
                {
                    "values": row,
                    "page": int(pages[index]) if index < len(pages) else group.last_page,
                    "confidence": 1.0,
                }
            )
        if not rows_payload:
            continue

        tables[table.name] = rows_payload
        settled.add(table.name)

        # The table is still open when gold rows exist beyond this group's last page.
        pages = row_pages.get(table.name, ())
        if pages and any(int(page) > group.last_page for page in pages):
            open_tables.append(table.name)

    target = json.dumps(
        {
            "fields": fields,
            "evidence": evidence,
            "tables": tables,
            "open_tables": open_tables,
        },
        indent=2,
        default=str,
    )
    return target, settled, open_tables


def build_examples(
    document: Document,
    schema: ExtractionSchema,
    gold: dict[str, Any],
    gold_evidence: dict[str, Any] | None = None,
    *,
    grouping: GroupingConfig | None = None,
    row_pages: dict[str, Sequence[int]] | None = None,
    include_empty: bool = False,
    max_chars_per_page: int = 6_000,
) -> list[TrainingExample]:
    """Turn one labelled document into page-group training examples.

    Args:
        document: The source document.
        schema: The extraction target.
        gold: The document-level gold record.
        gold_evidence: Per-key gold evidence with page numbers. Strongly recommended -
            without it, every scalar is attributed to the first group, which teaches
            the model nothing about where values live.
        grouping: Must match inference-time grouping, or training and serving diverge.
        row_pages: Per-table list of gold page numbers, one per row. This is what
            makes table continuation learnable.
        include_empty: Keep groups whose target extracts nothing. A minority of these
            is useful - it teaches the model to output an empty envelope rather than
            hallucinate - but too many teach it to say nothing.

    Returns:
        One example per page group, in document order.
    """
    gold_evidence = gold_evidence or {}
    groups = partition(document, grouping or GroupingConfig(), schema)
    examples: list[TrainingExample] = []
    state = CrossPageState(schema=schema, document_id=document.document_id)
    already: set[str] = set()

    for group in groups:
        target, settled, open_tables = build_target(
            schema, group, gold, gold_evidence, already, row_pages=row_pages
        )
        payload = json.loads(target)
        is_empty = not payload["fields"] and not payload["tables"]

        if not is_empty or include_empty:
            prompt = build_prompt(
                schema,
                group,
                state if group.group_index else None,
                max_chars_per_page=max_chars_per_page,
            )
            examples.append(
                TrainingExample(
                    document_id=document.document_id,
                    group_index=group.group_index,
                    page_numbers=group.page_numbers,
                    system=prompt.system,
                    user=prompt.user,
                    target=target,
                    image_paths=list(prompt.image_paths),
                )
            )

        # Advance the simulated state exactly as the pipeline would have.
        for key, value in payload["fields"].items():
            refs = [
                EvidenceRef(
                    page_number=int(claim.get("page", group.last_page)),
                    block_id=claim.get("block_id"),
                    quote=claim.get("quote", ""),
                    confidence=1.0,
                )
                for claim in payload["evidence"].get(key, [])
            ]
            state.update_field(
                key, value, group_index=group.group_index, confidence=1.0, evidence=refs
            )
        for table_name, rows in payload["tables"].items():
            state.append_rows(
                table_name,
                [row["values"] for row in rows],
                group_index=group.group_index,
                page_number=group.last_page,
                confidence=1.0,
            )
        for table in schema.tables:
            state.mark_table_open(table.name, open_=table.name in open_tables)

        already |= {
            key
            for key in settled
            if (spec := schema.field(key)) is None
            or (not spec.continues_across_pages and spec.cardinality is not Cardinality.MANY)
        }

    return examples


def build_corpus(
    labelled: Iterable[Any],
    schema: ExtractionSchema,
    *,
    grouping: GroupingConfig | None = None,
    include_empty_ratio: float = 0.1,
    seed: int = 13,
) -> list[TrainingExample]:
    """Build examples across a labelled corpus.

    ``include_empty_ratio`` keeps a small, random share of empty-target groups. Some
    are necessary - a model that has never seen a page with nothing on it will invent
    something for one - but a corpus dominated by them trains a model that shrugs.
    """
    rng = random.Random(seed)
    examples: list[TrainingExample] = []

    for item in labelled:
        document = getattr(item, "document", None) or item["document"]
        gold = getattr(item, "gold", None) if hasattr(item, "gold") else item.get("gold", {})
        gold_evidence = (
            getattr(item, "gold_evidence", None)
            if hasattr(item, "gold_evidence")
            else item.get("gold_evidence", {})
        )
        row_pages = getattr(item, "row_pages", None) or (
            item.get("row_pages") if isinstance(item, dict) else None
        )

        non_empty = build_examples(
            document, schema, gold, gold_evidence, grouping=grouping, row_pages=row_pages
        )
        examples.extend(non_empty)

        if include_empty_ratio > 0:
            everything = build_examples(
                document,
                schema,
                gold,
                gold_evidence,
                grouping=grouping,
                row_pages=row_pages,
                include_empty=True,
            )
            kept = {(e.document_id, e.group_index) for e in non_empty}
            empties = [e for e in everything if (e.document_id, e.group_index) not in kept]
            quota = int(len(non_empty) * include_empty_ratio)
            examples.extend(rng.sample(empties, min(quota, len(empties))))

    rng.shuffle(examples)
    return examples


def write_jsonl(examples: Sequence[TrainingExample], path: str | Path) -> int:
    """Write examples as JSONL. Returns the number written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), default=str) + "\n")
    return len(examples)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL training file."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split(
    examples: Sequence[TrainingExample],
    *,
    train_fraction: float = 0.9,
    seed: int = 13,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split by *document*, never by example.

    Splitting by example would put group 0 of a document in train and group 1 in
    validation, and the carry-over state would leak the answer straight across the
    boundary. Grouping by document is the only honest split for this data.
    """
    documents = sorted({example.document_id for example in examples})
    rng = random.Random(seed)
    rng.shuffle(documents)

    cut = int(len(documents) * train_fraction)
    train_ids = set(documents[:cut])
    train = [e for e in examples if e.document_id in train_ids]
    validation = [e for e in examples if e.document_id not in train_ids]
    return train, validation
