"""Validation-driven early exit.

Most of a long document is not about the fields you want. Once every required key is
filled, validates cleanly, and is backed by verified evidence, reading the remaining
twenty pages buys nothing - unless a table is still open, in which case stopping
would silently truncate it.

That caveat is the whole design. A naive "stop when the schema is satisfied" policy
looks excellent on header fields and quietly loses half a line-item table. So the
policy here refuses to stop while any table is mid-flight, and separately requires
consecutive unproductive groups before it accepts that the document has nothing left
to give.

Every decision is returned with its reason, so a run's stopping point is auditable
rather than mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from throughline.schema.spec import ExtractionSchema
from throughline.schema.validate import validate_record
from throughline.state.cross_page import CrossPageState


class ExitReason(str, Enum):
    """Why a run stopped."""

    SCHEMA_SATISFIED = "schema_satisfied"
    NO_NEW_INFORMATION = "no_new_information"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ALL_GROUPS_PROCESSED = "all_groups_processed"
    ERROR = "error"

    @property
    def is_early(self) -> bool:
        """True when the run stopped before reading every group."""
        return self in {
            ExitReason.SCHEMA_SATISFIED,
            ExitReason.NO_NEW_INFORMATION,
            ExitReason.BUDGET_EXHAUSTED,
        }


@dataclass(frozen=True)
class ExitDecision:
    """Whether to stop, and why."""

    should_stop: bool
    reason: ExitReason | None = None
    detail: str = ""

    def __bool__(self) -> bool:
        return self.should_stop


@dataclass
class EarlyExitPolicy:
    """Decide after each page group whether the run can stop.

    Args:
        enabled: Turn the whole policy off to force a full pass - which is what the
            accuracy-ceiling configuration does when measuring the cost of the speedup.
        min_groups: Never stop before this many groups, however satisfied the schema
            looks. Guards against a first page that happens to contain every header
            field while the tables live further in.
        require_valid: Only stop when the accumulated record is schema-valid.
        require_evidence: Only stop when every required key has at least one verified
            citation. A required field with no evidence is exactly the kind of value
            that turns out to be wrong.
        min_confidence: Mean confidence floor before stopping.
        patience: Consecutive groups that add nothing before declaring the document
            exhausted. 2 means one unproductive group is tolerated - overlap regions
            legitimately produce nothing new.
        respect_open_tables: Never stop while a table is mid-flight. Turning this off
            is how you reproduce the truncation bug it exists to prevent.
        max_groups: Hard bound on groups read, regardless of everything above.
    """

    enabled: bool = True
    min_groups: int = 1
    require_valid: bool = True
    require_evidence: bool = True
    min_confidence: float = 0.0
    patience: int = 2
    respect_open_tables: bool = True
    max_groups: int | None = None

    _unproductive_streak: int = field(default=0, init=False)
    _history: list[dict[str, Any]] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self._unproductive_streak = 0
        self._history.clear()

    def note_group(self, *, changed: bool) -> None:
        """Record whether the group just processed changed the state."""
        self._unproductive_streak = 0 if changed else self._unproductive_streak + 1

    def evaluate(
        self,
        state: CrossPageState,
        schema: ExtractionSchema,
        *,
        groups_processed: int,
        groups_total: int,
    ) -> ExitDecision:
        """Decide whether to stop after ``groups_processed`` groups."""
        if groups_processed >= groups_total:
            return ExitDecision(True, ExitReason.ALL_GROUPS_PROCESSED, "every group read")

        if self.max_groups is not None and groups_processed >= self.max_groups:
            return ExitDecision(
                True, ExitReason.BUDGET_EXHAUSTED, f"group budget of {self.max_groups} reached"
            )

        if not self.enabled:
            return ExitDecision(False)

        if groups_processed < self.min_groups:
            return ExitDecision(False, detail=f"below min_groups={self.min_groups}")

        if self.respect_open_tables and state.open_tables:
            self._record(groups_processed, False, "table still open")
            return ExitDecision(
                False, detail=f"tables still open: {sorted(state.open_tables)}"
            )

        if self._unproductive_streak >= self.patience:
            return ExitDecision(
                True,
                ExitReason.NO_NEW_INFORMATION,
                f"{self._unproductive_streak} consecutive groups added nothing",
            )

        missing = state.missing_required()
        if missing:
            self._record(groups_processed, False, f"missing {missing}")
            return ExitDecision(False, detail=f"required keys still missing: {missing}")

        record = state.to_record()
        if self.require_valid:
            _, report = validate_record(schema, record)
            if not report.is_valid:
                self._record(groups_processed, False, "record invalid")
                return ExitDecision(
                    False, detail=f"record not yet schema-valid: {report.summary()}"
                )

        if self.require_evidence:
            unsupported = [
                key
                for key in schema.required_keys
                if key in state.fields and not state.fields[key].evidence
            ]
            if unsupported:
                self._record(groups_processed, False, f"unsupported {unsupported}")
                return ExitDecision(
                    False, detail=f"required keys lack verified evidence: {unsupported}"
                )

        confidence = state.mean_confidence()
        if confidence < self.min_confidence:
            self._record(groups_processed, False, f"confidence {confidence:.2f}")
            return ExitDecision(
                False,
                detail=f"mean confidence {confidence:.2f} below floor {self.min_confidence:.2f}",
            )

        self._record(groups_processed, True, "schema satisfied")
        return ExitDecision(
            True,
            ExitReason.SCHEMA_SATISFIED,
            f"all required keys present, valid and evidenced after {groups_processed} group(s)",
        )

    def _record(self, groups_processed: int, stop: bool, detail: str) -> None:
        self._history.append(
            {"groups_processed": groups_processed, "stop": stop, "detail": detail}
        )

    @property
    def history(self) -> list[dict[str, Any]]:
        """Every decision made during the run, for audit and debugging."""
        return list(self._history)


ACCURACY_CEILING = EarlyExitPolicy(enabled=False)
"""Read every group. The configuration used to measure peak accuracy."""

BALANCED = EarlyExitPolicy(
    enabled=True, min_groups=2, require_valid=True, require_evidence=True, patience=2
)
"""The production default: stop when the schema is satisfied and evidenced."""

AGGRESSIVE = EarlyExitPolicy(
    enabled=True, min_groups=1, require_valid=True, require_evidence=False, patience=1
)
"""Lower latency, weaker guarantees. Appropriate for triage, not for a system of record."""
