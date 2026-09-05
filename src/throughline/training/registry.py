"""Model registry and the promotion gate.

Training produces an adapter. That is not the same as having a model worth deploying,
and the distance between the two is where most model-quality incidents live. This
module is the decision layer: it records what was trained, from what, scoring what,
and it refuses to promote a candidate that is not demonstrably better than the model
currently serving traffic.

Three properties matter, and each exists because of a specific way this goes wrong.

**A mean can improve while a field collapses.** Weighted F1 is support-weighted, so a
candidate can gain half a point overall while `total_amount` — one field among eleven,
and the one finance actually reads — drops twenty. :class:`PromotionGate` checks
**per-key regression** independently of the aggregate, which is the check that catches
this.

**A better model on a different corpus is not a comparison.** Every
:class:`ModelCard` carries a `corpus_fingerprint`. Comparing two cards scored on
different data is refused rather than silently allowed.

**Rollback must be a lookup, not an archaeology exercise.** The registry keeps full
history with stages, so "what was serving last Tuesday, and what did it score?" is one
call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class Stage(str, Enum):
    """Where a model sits in its lifecycle."""

    CANDIDATE = "candidate"
    """Trained and scored, not yet judged."""

    STAGING = "staging"
    """Passed the gate; deployed to a non-production endpoint."""

    CHAMPION = "champion"
    """Currently serving."""

    ARCHIVED = "archived"
    """Superseded or rejected. Kept, because rollback targets live here."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    """Short SHA of the working tree, or ``"unknown"`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment-gated
        return "unknown"


def corpus_fingerprint(document_ids: Iterable[str]) -> str:
    """Order-independent fingerprint of an evaluation corpus.

    Two runs scored on the same documents produce the same fingerprint regardless of
    iteration order; adding or removing one document changes it. This is what makes
    "is this comparison valid?" a mechanical question.
    """
    digest = hashlib.sha256()
    for document_id in sorted(set(document_ids)):
        digest.update(document_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


@dataclass
class ModelCard:
    """Everything needed to judge, deploy, or reproduce one trained model."""

    model_id: str
    base_model: str
    adapter_uri: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_key_f1: dict[str, float] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    corpus_fingerprint: str = ""
    corpus_size: int = 0
    schema: str = ""
    stage: Stage = Stage.CANDIDATE
    created_at: str = field(default_factory=_now)
    git_sha: str = field(default_factory=_git_sha)
    mlflow_run_id: str = ""
    notes: list[str] = field(default_factory=list)

    def metric(self, name: str, default: float = 0.0) -> float:
        return float(self.metrics.get(name, default))

    def config_hash(self) -> str:
        """Hash of the training configuration, to spot accidental duplicate runs."""
        payload = json.dumps(self.training_config, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model": self.base_model,
            "adapter_uri": self.adapter_uri,
            "metrics": self.metrics,
            "per_key_f1": self.per_key_f1,
            "training_config": self.training_config,
            "corpus_fingerprint": self.corpus_fingerprint,
            "corpus_size": self.corpus_size,
            "schema": self.schema,
            "stage": self.stage.value,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "mlflow_run_id": self.mlflow_run_id,
            "config_hash": self.config_hash(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelCard:
        card = cls(
            model_id=payload["model_id"],
            base_model=payload.get("base_model", ""),
            adapter_uri=payload.get("adapter_uri", ""),
            metrics={k: float(v) for k, v in payload.get("metrics", {}).items()},
            per_key_f1={k: float(v) for k, v in payload.get("per_key_f1", {}).items()},
            training_config=payload.get("training_config", {}),
            corpus_fingerprint=payload.get("corpus_fingerprint", ""),
            corpus_size=int(payload.get("corpus_size", 0)),
            schema=payload.get("schema", ""),
            stage=Stage(payload.get("stage", "candidate")),
            created_at=payload.get("created_at", _now()),
            git_sha=payload.get("git_sha", "unknown"),
            mlflow_run_id=payload.get("mlflow_run_id", ""),
        )
        card.notes = list(payload.get("notes", []))
        return card

    @classmethod
    def from_evaluation(
        cls,
        model_id: str,
        run: Any,
        *,
        base_model: str,
        adapter_uri: str,
        training_config: dict[str, Any] | None = None,
        schema: str = "",
        mlflow_run_id: str = "",
    ) -> ModelCard:
        """Build a card from an :class:`~throughline.evaluation.harness.EvaluationRun`."""
        report = run.report
        return cls(
            model_id=model_id,
            base_model=base_model,
            adapter_uri=adapter_uri,
            metrics=report.headline(),
            per_key_f1={key: score.f1 for key, score in report.per_key.items()},
            training_config=training_config or {},
            corpus_fingerprint=corpus_fingerprint(r.document_id for r in run.results),
            corpus_size=report.documents,
            schema=schema,
            mlflow_run_id=mlflow_run_id,
        )


# ── promotion gate ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GateCheck:
    """One check, its verdict, and enough detail to argue with it."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.blocking else "WARN")
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class GateDecision:
    """Whether a candidate may be promoted, and every check behind that."""

    promote: bool
    checks: list[GateCheck] = field(default_factory=list)
    candidate_id: str = ""
    champion_id: str = ""

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed and check.blocking]

    @property
    def warnings(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed and not check.blocking]

    def __bool__(self) -> bool:
        return self.promote

    def report(self) -> str:
        header = (
            f"{'PROMOTE' if self.promote else 'HOLD'}: "
            f"{self.candidate_id or '(candidate)'} vs {self.champion_id or '(no champion)'}"
        )
        return "\n".join([header, *(f"  {check}" for check in self.checks)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "candidate_id": self.candidate_id,
            "champion_id": self.champion_id,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "blocking": c.blocking,
                }
                for c in self.checks
            ],
        }


@dataclass
class PromotionGate:
    """Decide whether a candidate should replace the current champion.

    Args:
        primary_metric: The metric a candidate must improve to earn promotion.
        min_improvement: How much it must improve by. Set above measurement noise;
            promoting on a 0.001 difference is promoting on nothing.
        guarded_metrics: Metrics that must not regress beyond ``metric_tolerance``,
            whatever the primary metric does. `citation_precision` belongs here: a
            model that extracts better but grounds worse is not an upgrade for a
            system whose premise is groundedness.
        metric_tolerance: Allowed regression on a guarded metric.
        per_key_tolerance: Allowed regression on any individual field's F1. This is
            the check that catches an improved average hiding a collapsed field.
        absolute_floors: Hard minimums a candidate must clear regardless of the
            champion — the floor below which the system is not fit to serve.
        require_same_corpus: Refuse to compare cards scored on different corpora.
        min_corpus_size: Refuse to promote on an evaluation too small to mean anything.
    """

    primary_metric: str = "weighted_f1"
    min_improvement: float = 0.005
    guarded_metrics: tuple[str, ...] = ("schema_valid_rate", "citation_precision")
    metric_tolerance: float = 0.01
    per_key_tolerance: float = 0.05
    absolute_floors: dict[str, float] = field(
        default_factory=lambda: {"schema_valid_rate": 0.95, "citation_precision": 0.90}
    )
    require_same_corpus: bool = True
    min_corpus_size: int = 25

    def evaluate(self, candidate: ModelCard, champion: ModelCard | None) -> GateDecision:
        """Judge a candidate. ``champion=None`` means nothing is serving yet."""
        checks: list[GateCheck] = []

        checks.append(
            GateCheck(
                "corpus_size",
                candidate.corpus_size >= self.min_corpus_size,
                f"scored on {candidate.corpus_size} documents "
                f"(minimum {self.min_corpus_size})",
            )
        )

        for name, floor in sorted(self.absolute_floors.items()):
            value = candidate.metric(name)
            checks.append(
                GateCheck(
                    f"floor:{name}",
                    value >= floor,
                    f"{value:.4f} vs floor {floor:.4f}",
                )
            )

        if champion is None:
            checks.append(
                GateCheck("baseline", True, "no champion; first model to pass the floors")
            )
            return GateDecision(
                promote=all(c.passed for c in checks if c.blocking),
                checks=checks,
                candidate_id=candidate.model_id,
            )

        if self.require_same_corpus:
            same = candidate.corpus_fingerprint == champion.corpus_fingerprint
            checks.append(
                GateCheck(
                    "same_corpus",
                    same,
                    "candidate and champion scored on the same corpus"
                    if same
                    else f"different corpora ({candidate.corpus_fingerprint} vs "
                    f"{champion.corpus_fingerprint}) — the comparison is not valid",
                )
            )

        delta = candidate.metric(self.primary_metric) - champion.metric(self.primary_metric)
        checks.append(
            GateCheck(
                f"improvement:{self.primary_metric}",
                delta >= self.min_improvement,
                f"{champion.metric(self.primary_metric):.4f} → "
                f"{candidate.metric(self.primary_metric):.4f} "
                f"({delta:+.4f}, need ≥ {self.min_improvement:+.4f})",
            )
        )

        for name in self.guarded_metrics:
            regression = champion.metric(name) - candidate.metric(name)
            checks.append(
                GateCheck(
                    f"guard:{name}",
                    regression <= self.metric_tolerance,
                    f"{champion.metric(name):.4f} → {candidate.metric(name):.4f} "
                    f"({-regression:+.4f}, tolerance {self.metric_tolerance:.4f})",
                )
            )

        regressed = self._per_key_regressions(candidate, champion)
        checks.append(
            GateCheck(
                "per_key_regression",
                not regressed,
                "no field regressed beyond tolerance"
                if not regressed
                else "; ".join(
                    f"{key} {before:.3f} → {after:.3f} ({after - before:+.3f})"
                    for key, before, after in regressed
                ),
            )
        )

        untested = sorted(set(champion.per_key_f1) - set(candidate.per_key_f1))
        if untested:
            checks.append(
                GateCheck(
                    "key_coverage",
                    False,
                    f"champion scored keys the candidate did not: {untested}",
                    blocking=False,
                )
            )

        return GateDecision(
            promote=all(check.passed for check in checks if check.blocking),
            checks=checks,
            candidate_id=candidate.model_id,
            champion_id=champion.model_id,
        )

    def _per_key_regressions(
        self, candidate: ModelCard, champion: ModelCard
    ) -> list[tuple[str, float, float]]:
        regressions = []
        for key, before in champion.per_key_f1.items():
            if key not in candidate.per_key_f1:
                continue
            after = candidate.per_key_f1[key]
            if before - after > self.per_key_tolerance:
                regressions.append((key, before, after))
        return sorted(regressions, key=lambda item: item[2] - item[1])


STRICT = PromotionGate(min_improvement=0.01, metric_tolerance=0.005, per_key_tolerance=0.02)
"""For a system of record. A candidate must clearly win and regress almost nothing."""

PERMISSIVE = PromotionGate(min_improvement=0.001, metric_tolerance=0.03, per_key_tolerance=0.10)
"""For early iteration, when the champion is weak and movement matters more than safety."""


# ── registry ──────────────────────────────────────────────────────────────
@dataclass
class ModelRegistry:
    """A JSON-backed registry of model cards, with stages and history.

    Deliberately file-backed rather than a service. The registry's job is to be the
    single answer to "what is serving, and what did it score?", and a JSON file in
    version control answers that from a laptop, from CI, and from a SageMaker job
    without anything needing to be up. Point ``path`` at S3-synced storage for a team.
    """

    path: str | Path = "results/model_registry.json"

    def _load(self) -> list[ModelCard]:
        file = Path(self.path)
        if not file.exists():
            return []
        payload = json.loads(file.read_text(encoding="utf-8"))
        return [ModelCard.from_dict(raw) for raw in payload.get("models", [])]

    def _save(self, cards: Sequence[ModelCard]) -> None:
        file = Path(self.path)
        file.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {"updated_at": _now(), "models": [card.to_dict() for card in cards]},
            indent=2,
        )
        temporary = file.with_suffix(".tmp")
        temporary.write_text(body + "\n", encoding="utf-8")
        temporary.replace(file)

    # ── reads ─────────────────────────────────────────────────────────
    def all(self) -> list[ModelCard]:
        """Every card, newest first."""
        return sorted(self._load(), key=lambda card: card.created_at, reverse=True)

    def get(self, model_id: str) -> ModelCard | None:
        return next((card for card in self._load() if card.model_id == model_id), None)

    def champion(self, schema: str | None = None) -> ModelCard | None:
        """The model currently serving, optionally for one schema."""
        for card in self._load():
            if card.stage is not Stage.CHAMPION:
                continue
            if schema is None or card.schema == schema:
                return card
        return None

    def by_stage(self, stage: Stage) -> list[ModelCard]:
        return [card for card in self.all() if card.stage is stage]

    # ── writes ────────────────────────────────────────────────────────
    def register(self, card: ModelCard) -> ModelCard:
        """Add a candidate. Model ids are unique; re-registering replaces."""
        cards = [existing for existing in self._load() if existing.model_id != card.model_id]
        cards.append(card)
        self._save(cards)
        LOGGER.info("Registered %s (%s)", card.model_id, card.stage.value)
        return card

    def promote(
        self, model_id: str, gate: PromotionGate | None = None, *, force: bool = False
    ) -> GateDecision:
        """Promote a candidate to champion, if the gate allows it.

        The outgoing champion is archived rather than deleted — it is the rollback
        target, and a registry that cannot roll back is a list.
        """
        cards = self._load()
        candidate = next((card for card in cards if card.model_id == model_id), None)
        if candidate is None:
            raise KeyError(f"Unknown model: {model_id}")

        champion = next(
            (
                card
                for card in cards
                if card.stage is Stage.CHAMPION and card.schema == candidate.schema
            ),
            None,
        )

        gate = gate or PromotionGate()
        decision = gate.evaluate(candidate, champion)

        if not decision.promote and not force:
            # Debug, not info: the caller receives the decision and is responsible for
            # presenting it. Logging the full report here as well duplicates it in the CLI.
            LOGGER.debug("Promotion held for %s:\n%s", model_id, decision.report())
            return decision

        if force and not decision.promote:
            candidate.notes.append(
                f"force-promoted despite {len(decision.failures)} failed check(s) at {_now()}"
            )
            LOGGER.warning("Force-promoting %s over a failed gate", model_id)

        for card in cards:
            if card is champion:
                card.stage = Stage.ARCHIVED
                card.notes.append(f"superseded by {candidate.model_id} at {_now()}")
        candidate.stage = Stage.CHAMPION

        self._save(cards)
        decision = GateDecision(
            promote=True,
            checks=decision.checks,
            candidate_id=candidate.model_id,
            champion_id=champion.model_id if champion else "",
        )
        return decision

    def rollback(self, schema: str | None = None) -> ModelCard | None:
        """Restore the most recently archived champion. Returns the restored card."""
        cards = self._load()
        current = next(
            (
                card
                for card in cards
                if card.stage is Stage.CHAMPION and (schema is None or card.schema == schema)
            ),
            None,
        )
        archived = sorted(
            (
                card
                for card in cards
                if card.stage is Stage.ARCHIVED and (schema is None or card.schema == schema)
            ),
            key=lambda card: card.created_at,
            reverse=True,
        )
        if not archived:
            LOGGER.warning("No archived model to roll back to.")
            return None

        target = archived[0]
        if current is not None:
            current.stage = Stage.ARCHIVED
            current.notes.append(f"rolled back at {_now()}")
        target.stage = Stage.CHAMPION
        target.notes.append(f"restored by rollback at {_now()}")

        self._save(cards)
        LOGGER.info("Rolled back to %s", target.model_id)
        return target

    def set_stage(self, model_id: str, stage: Stage) -> ModelCard:
        cards = self._load()
        card = next((item for item in cards if item.model_id == model_id), None)
        if card is None:
            raise KeyError(f"Unknown model: {model_id}")
        card.stage = stage
        self._save(cards)
        return card

    # ── presentation ──────────────────────────────────────────────────
    def table(self) -> str:
        """The registry as a fixed-width table, for terminals and CI logs."""
        cards = self.all()
        if not cards:
            return "(registry is empty)"

        width = max(len(card.model_id) for card in cards) + 2
        lines = [
            f"{'model':<{width}}{'stage':<12}{'F1':>8}{'valid':>8}{'cite':>8}{'docs':>7}  created"
        ]
        lines.append("-" * (width + 55))
        for card in cards:
            lines.append(
                f"{card.model_id:<{width}}{card.stage.value:<12}"
                f"{card.metric('weighted_f1'):>8.3f}"
                f"{card.metric('schema_valid_rate'):>8.3f}"
                f"{card.metric('citation_precision'):>8.3f}"
                f"{card.corpus_size:>7d}  {card.created_at[:10]}"
            )
        return "\n".join(lines)
