"""The extraction orchestrator.

This is where the pieces become a system. For one document it:

1. partitions the pages into bounded, overlapping groups;
2. optionally reorders those groups so the most promising are read first;
3. for each group - assembles a prompt carrying the cross-page state, calls the
   backend (through the prompt cache), parses and repairs the output, verifies every
   citation, and merges the result into state;
4. asks the early-exit policy whether the schema is satisfied;
5. validates the accumulated record and returns it with its full provenance.

Two decisions in here are worth naming because they are where most of the behaviour
comes from.

**State goes into the prompt, not just out of it.** Each group is told what earlier
groups established and what is still missing. That is what makes a bounded window
behave like a whole-document read.

**Nothing is trusted without attribution.** A value whose citation does not resolve
is kept but marked unverified, and unverified required fields block early exit. The
system would rather read three more pages than emit a confident value it cannot
point at.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from throughline.attribution.evidence import (
    AttributionStats,
    attribute_all,
)
from throughline.caching.store import CachedBackend, PromptCache
from throughline.decoding.constrained import ParsedEnvelope, ParseError, parse_envelope
from throughline.grouping.page_groups import (
    GroupingConfig,
    PageGroup,
    partition,
)
from throughline.ingest.layout import Document
from throughline.models.base import GenerationConfig, GenerationResult
from throughline.pipeline.early_exit import EarlyExitPolicy, ExitDecision, ExitReason
from throughline.prompting.templates import build_prompt, build_repair_prompt
from throughline.retrieval.relevant_pages import RelevantPageRetriever
from throughline.schema.spec import ExtractionSchema
from throughline.schema.validate import ValidationReport, validate_record
from throughline.state.cross_page import CrossPageState, EvidenceRef

LOGGER = logging.getLogger(__name__)


@dataclass
class GroupTrace:
    """What happened while processing one page group."""

    group_index: int
    page_numbers: tuple[int, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    cached: bool = False
    fields_added: int = 0
    rows_added: int = 0
    citations_claimed: int = 0
    citations_verified: int = 0
    repairs: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def changed_state(self) -> bool:
        return bool(self.fields_added or self.rows_added)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_index": self.group_index,
            "page_numbers": list(self.page_numbers),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": round(self.latency_seconds, 4),
            "cached": self.cached,
            "fields_added": self.fields_added,
            "rows_added": self.rows_added,
            "citations_claimed": self.citations_claimed,
            "citations_verified": self.citations_verified,
            "repairs": list(self.repairs),
            "error": self.error,
        }


@dataclass
class ExtractionResult:
    """Everything one document produced."""

    document_id: str
    schema_name: str
    record: dict[str, Any]
    validation: ValidationReport
    state: CrossPageState
    traces: list[GroupTrace] = field(default_factory=list)
    exit_reason: ExitReason | None = None
    exit_detail: str = ""
    groups_total: int = 0
    wall_seconds: float = 0.0
    attribution: AttributionStats = field(default_factory=AttributionStats)

    @property
    def groups_processed(self) -> int:
        return len(self.traces)

    @property
    def pages_read(self) -> int:
        return len({page for trace in self.traces for page in trace.page_numbers})

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid

    @property
    def total_tokens(self) -> int:
        return sum(trace.prompt_tokens + trace.completion_tokens for trace in self.traces)

    @property
    def groups_skipped(self) -> int:
        return max(self.groups_total - self.groups_processed, 0)

    def evidence_for(self, key: str) -> list[EvidenceRef]:
        entry = self.state.fields.get(key)
        if entry is not None:
            return entry.evidence
        return [ref for row in self.state.tables.get(key, []) for ref in row.evidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "schema": self.schema_name,
            "record": self.record,
            "validation": self.validation.to_dict(),
            "evidence": self.state.evidence_map(),
            "exit": {
                "reason": self.exit_reason.value if self.exit_reason else None,
                "detail": self.exit_detail,
                "groups_processed": self.groups_processed,
                "groups_total": self.groups_total,
                "groups_skipped": self.groups_skipped,
                "pages_read": self.pages_read,
            },
            "cost": {
                "wall_seconds": round(self.wall_seconds, 4),
                "total_tokens": self.total_tokens,
            },
            "attribution": self.attribution.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "notes": self.state.notes,
        }

    def summary(self) -> str:
        parts = [
            f"{self.document_id}: {self.validation.summary()}",
            f"{self.groups_processed}/{self.groups_total} groups",
            f"{self.pages_read} pages",
            f"{self.attribution.citation_precision:.0%} citations verified",
            f"{self.wall_seconds:.2f}s",
        ]
        if self.exit_reason and self.exit_reason.is_early:
            parts.append(f"early exit ({self.exit_reason.value})")
        return " · ".join(parts)


@dataclass
class PipelineConfig:
    """Everything tunable about one extraction run."""

    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    early_exit: EarlyExitPolicy = field(default_factory=EarlyExitPolicy)
    use_retrieval: bool = True
    """Reorder groups by predicted relevance to the fields still missing."""

    max_chars_per_page: int = 6_000
    repair_attempts: int = 1
    """Cheap second passes asking the model to fix a schema-invalid output."""

    prompt_cache: PromptCache | None = None
    fail_fast: bool = False
    """Raise on the first group error instead of recording it and continuing."""


class ExtractionPipeline:
    """Run schema-constrained extraction over long documents."""

    def __init__(
        self,
        backend: Any,
        schema: ExtractionSchema,
        config: PipelineConfig | None = None,
    ) -> None:
        self.schema = schema
        self.config = config or PipelineConfig()
        self.backend = (
            CachedBackend(backend, self.config.prompt_cache)
            if self.config.prompt_cache is not None
            else backend
        )

    # ── public API ────────────────────────────────────────────────────────
    def run(self, document: Document) -> ExtractionResult:
        """Extract one document."""
        started = time.perf_counter()

        groups = partition(document, self.config.grouping, self.schema)
        state = CrossPageState(schema=self.schema, document_id=document.document_id)
        policy = self.config.early_exit
        policy.reset()

        traces: list[GroupTrace] = []
        stats = AttributionStats()
        decision = ExitDecision(False)

        ordered = self._order_groups(document, groups, state)

        for position, group in enumerate(ordered, start=1):
            trace = self._process_group(document, group, state, stats)
            traces.append(trace)
            policy.note_group(changed=trace.changed_state)

            decision = policy.evaluate(
                state,
                self.schema,
                groups_processed=position,
                groups_total=len(groups),
            )
            LOGGER.debug(
                "group %s: +%s fields +%s rows | %s",
                group.group_index,
                trace.fields_added,
                trace.rows_added,
                decision.detail or ("stop" if decision.should_stop else "continue"),
            )
            if decision.should_stop:
                break

            if self.config.use_retrieval and state.missing_required():
                ordered = self._reorder_remaining(document, ordered, position, state)

        record, report = validate_record(self.schema, state.to_record())

        return ExtractionResult(
            document_id=document.document_id,
            schema_name=self.schema.name,
            record=record,
            validation=report,
            state=state,
            traces=traces,
            exit_reason=decision.reason or ExitReason.ALL_GROUPS_PROCESSED,
            exit_detail=decision.detail,
            groups_total=len(groups),
            wall_seconds=time.perf_counter() - started,
            attribution=stats,
        )

    def run_batch(self, documents: Sequence[Document]) -> list[ExtractionResult]:
        """Extract a corpus, isolating per-document failures."""
        results: list[ExtractionResult] = []
        for document in documents:
            try:
                results.append(self.run(document))
            except Exception as exc:  # noqa: BLE001 - one bad document must not stop a batch
                if self.config.fail_fast:
                    raise
                LOGGER.exception("Document %s failed", document.document_id)
                empty = CrossPageState(schema=self.schema, document_id=document.document_id)
                _, report = validate_record(self.schema, {})
                results.append(
                    ExtractionResult(
                        document_id=document.document_id,
                        schema_name=self.schema.name,
                        record={},
                        validation=report,
                        state=empty,
                        exit_reason=ExitReason.ERROR,
                        exit_detail=str(exc),
                    )
                )
        return results

    # ── group ordering ────────────────────────────────────────────────────
    def _order_groups(
        self, document: Document, groups: list[PageGroup], state: CrossPageState
    ) -> list[PageGroup]:
        """Read in page order by default; by predicted relevance when retrieval is on.

        Page order is not merely a fallback - it is required for correctness whenever
        a table spans groups, because rows only accumulate correctly in document
        order. So relevance reordering is skipped for schemas that declare tables.
        """
        if not self.config.use_retrieval or not groups or self.schema.tables:
            return list(groups)

        retriever = RelevantPageRetriever(document, self.schema)
        targets = state.missing_required() or list(self.schema.all_keys)
        return [group for group, _ in retriever.rank_groups(groups, targets)]

    def _reorder_remaining(
        self,
        document: Document,
        ordered: list[PageGroup],
        position: int,
        state: CrossPageState,
    ) -> list[PageGroup]:
        """Re-rank the not-yet-read tail against what is still missing."""
        if self.schema.tables:
            return ordered

        head, tail = ordered[:position], ordered[position:]
        if len(tail) < 2:
            return ordered

        retriever = RelevantPageRetriever(document, self.schema)
        ranked = [group for group, _ in retriever.rank_groups(tail, state.missing_required())]
        return head + ranked

    # ── one group ─────────────────────────────────────────────────────────
    def _process_group(
        self,
        document: Document,
        group: PageGroup,
        state: CrossPageState,
        stats: AttributionStats,
    ) -> GroupTrace:
        trace = GroupTrace(group_index=group.group_index, page_numbers=group.page_numbers)

        prompt = build_prompt(
            self.schema,
            group,
            state if state.groups_processed else None,
            max_chars_per_page=self.config.max_chars_per_page,
            focus_fields=state.missing_required() or None,
        )

        try:
            result = self.backend.generate(prompt, self.config.generation)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            if self.config.fail_fast:
                raise
            trace.error = f"backend error: {exc}"
            LOGGER.warning("Group %s backend error: %s", group.group_index, exc)
            return trace

        trace.prompt_tokens = result.prompt_tokens
        trace.completion_tokens = result.completion_tokens
        trace.latency_seconds = result.latency_seconds
        trace.cached = result.cached

        envelope = self._parse_with_repair(result, trace)
        if envelope is None:
            return trace

        trace.repairs = list(envelope.repairs)
        self._merge(document, group, state, envelope, trace, stats)
        state.record_group(group.group_index, group.page_numbers)
        return trace

    def _parse_with_repair(
        self, result: GenerationResult, trace: GroupTrace
    ) -> ParsedEnvelope | None:
        """Parse the output, asking the model to fix it if it cannot be recovered."""
        try:
            return parse_envelope(result.text, self.schema)
        except ParseError as parse_error:
            # Python unbinds an `except ... as name` at the end of the block, so the
            # failure has to be carried out explicitly to survive into the retry loop.
            failure: Exception = parse_error
            LOGGER.debug("Group %s parse failed: %s", trace.group_index, failure)

        for attempt in range(self.config.repair_attempts):
            repair_prompt = build_repair_prompt(self.schema, result.text, [str(failure)])
            try:
                repaired = self.backend.generate(repair_prompt, self.config.generation)
                envelope = parse_envelope(repaired.text, self.schema)
                envelope.repairs.append(f"model repair pass {attempt + 1}")
                trace.completion_tokens += repaired.completion_tokens
                return envelope
            except Exception as repair_error:  # noqa: BLE001 - recorded below
                failure = repair_error
                continue

        trace.error = f"unparseable output: {failure}"
        return None

    def _merge(
        self,
        document: Document,
        group: PageGroup,
        state: CrossPageState,
        envelope: ParsedEnvelope,
        trace: GroupTrace,
        stats: AttributionStats,
    ) -> None:
        """Verify citations and fold one group's output into cross-page state."""
        pages = list(group.pages)

        for name, value in envelope.fields.items():
            claims = envelope.evidence.get(name, [])
            trace.citations_claimed += len(claims)
            refs, results = attribute_all(claims, value, pages, document=document)
            for outcome in results:
                stats.record(outcome)
            trace.citations_verified += len(refs)

            confidence = max((ref.confidence for ref in refs), default=0.0)
            if not refs and claims:
                # Keep the value, but it earns no citation and a confidence penalty.
                confidence = 0.2 * max(
                    (float(claim.get("confidence", 0.0) or 0.0) for claim in claims),
                    default=0.0,
                )
                state.notes.append(
                    f"group {group.group_index}: {name!r} kept without verified evidence"
                )

            if state.update_field(
                name, value, group_index=group.group_index, confidence=confidence, evidence=refs
            ):
                trace.fields_added += 1

        for table_name, rows in envelope.tables.items():
            prepared: list[dict[str, Any]] = []
            refs: list[EvidenceRef] = []
            page_number = group.last_page

            for row in rows:
                values = row.get("values", {})
                if not isinstance(values, dict) or not values:
                    continue
                prepared.append(values)
                claim = {
                    "block_id": row.get("block_id"),
                    "quote": row.get("quote", ""),
                    "confidence": row.get("confidence", 0.0),
                }
                trace.citations_claimed += 1
                row_refs, results = attribute_all([claim], values, pages, document=document)
                for outcome in results:
                    stats.record(outcome)
                trace.citations_verified += len(row_refs)
                refs.extend(row_refs)
                if row.get("page"):
                    page_number = int(row["page"])

            added = state.append_rows(
                table_name,
                prepared,
                group_index=group.group_index,
                page_number=page_number,
                evidence=refs,
                confidence=max((ref.confidence for ref in refs), default=0.0),
            )
            trace.rows_added += added

        for table in self.schema.tables:
            state.mark_table_open(table.name, open_=table.name in envelope.open_tables)
