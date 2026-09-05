"""The model registry, the promotion gate, and the retraining pipeline plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from throughline.training.pipeline import (
    PipelineConfig,
    build_plan,
    export_definition,
    render_plan,
    validate_plan,
)
from throughline.training.registry import (
    STRICT,
    GateDecision,
    ModelCard,
    ModelRegistry,
    PromotionGate,
    Stage,
    corpus_fingerprint,
)

CORPUS = corpus_fingerprint([f"doc_{n:03d}" for n in range(40)])


def card(
    model_id: str,
    *,
    f1: float = 0.90,
    valid: float = 0.98,
    cite: float = 0.94,
    per_key: dict[str, float] | None = None,
    stage: Stage = Stage.CANDIDATE,
    fingerprint: str = CORPUS,
    size: int = 40,
) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_uri=f"s3://bucket/adapters/{model_id}",
        metrics={
            "weighted_f1": f1,
            "schema_valid_rate": valid,
            "citation_precision": cite,
        },
        per_key_f1=per_key or {"invoice_number": 0.99, "total_amount": 0.95, "line_items": 0.88},
        corpus_fingerprint=fingerprint,
        corpus_size=size,
        schema="invoice",
        stage=stage,
    )


class TestCorpusFingerprint:
    def test_order_does_not_matter(self) -> None:
        assert corpus_fingerprint(["b", "a", "c"]) == corpus_fingerprint(["a", "c", "b"])

    def test_duplicates_do_not_matter(self) -> None:
        assert corpus_fingerprint(["a", "a", "b"]) == corpus_fingerprint(["a", "b"])

    def test_a_different_document_changes_it(self) -> None:
        assert corpus_fingerprint(["a", "b"]) != corpus_fingerprint(["a", "b", "c"])


class TestPromotionGate:
    def test_first_model_passes_on_the_floors_alone(self) -> None:
        decision = PromotionGate().evaluate(card("v1"), None)
        assert decision.promote
        assert any(check.name == "baseline" for check in decision.checks)

    def test_first_model_still_has_to_clear_the_floors(self) -> None:
        decision = PromotionGate().evaluate(card("v1", valid=0.50), None)
        assert not decision.promote
        assert any(c.name == "floor:schema_valid_rate" for c in decision.failures)

    def test_a_clear_improvement_promotes(self) -> None:
        decision = PromotionGate().evaluate(card("v2", f1=0.93), card("v1", f1=0.90))
        assert decision.promote

    def test_a_negligible_improvement_does_not(self) -> None:
        """Promoting on 0.001 is promoting on measurement noise."""
        decision = PromotionGate().evaluate(card("v2", f1=0.9005), card("v1", f1=0.90))
        assert not decision.promote
        assert any("improvement" in check.name for check in decision.failures)

    def test_a_regression_does_not_promote(self) -> None:
        decision = PromotionGate().evaluate(card("v2", f1=0.85), card("v1", f1=0.90))
        assert not decision.promote

    def test_a_guarded_metric_regression_blocks_a_better_f1(self) -> None:
        """A model that extracts better but grounds worse is not an upgrade."""
        decision = PromotionGate().evaluate(
            card("v2", f1=0.95, cite=0.80), card("v1", f1=0.90, cite=0.94)
        )
        assert not decision.promote
        assert any(c.name == "guard:citation_precision" for c in decision.failures)

    def test_a_collapsed_field_blocks_a_better_average(self) -> None:
        """The check that exists because a support-weighted mean can hide this."""
        decision = PromotionGate().evaluate(
            card("v2", f1=0.93, per_key={"invoice_number": 0.99, "total_amount": 0.40,
                                         "line_items": 0.95}),
            card("v1", f1=0.90),
        )
        assert not decision.promote
        failure = next(c for c in decision.failures if c.name == "per_key_regression")
        assert "total_amount" in failure.detail

    def test_a_small_per_key_dip_is_tolerated(self) -> None:
        decision = PromotionGate().evaluate(
            card("v2", f1=0.93, per_key={"invoice_number": 0.99, "total_amount": 0.93,
                                         "line_items": 0.90}),
            card("v1", f1=0.90),
        )
        assert decision.promote

    def test_different_corpora_is_not_a_comparison(self) -> None:
        decision = PromotionGate().evaluate(
            card("v2", f1=0.99, fingerprint="deadbeef"), card("v1", f1=0.90)
        )
        assert not decision.promote
        assert any(c.name == "same_corpus" for c in decision.failures)

    def test_too_small_an_evaluation_blocks_promotion(self) -> None:
        decision = PromotionGate().evaluate(card("v2", f1=0.99, size=3), card("v1", f1=0.90))
        assert not decision.promote
        assert any(c.name == "corpus_size" for c in decision.failures)

    def test_an_untested_key_is_a_warning_not_a_block(self) -> None:
        decision = PromotionGate().evaluate(
            card("v2", f1=0.93, per_key={"invoice_number": 0.99}), card("v1", f1=0.90)
        )
        assert decision.promote
        assert any(c.name == "key_coverage" for c in decision.warnings)

    def test_strict_profile_is_harder_to_pass(self) -> None:
        candidate, champion = card("v2", f1=0.907), card("v1", f1=0.90)
        assert PromotionGate().evaluate(candidate, champion).promote
        assert not STRICT.evaluate(candidate, champion).promote

    def test_the_report_names_every_check(self) -> None:
        decision = PromotionGate().evaluate(card("v2", f1=0.85), card("v1", f1=0.90))
        report = decision.report()
        assert "HOLD" in report
        assert "improvement:weighted_f1" in report

    def test_decision_is_truthy_only_when_promoting(self) -> None:
        assert bool(GateDecision(promote=True)) is True
        assert bool(GateDecision(promote=False)) is False


class TestModelRegistry:
    @pytest.fixture
    def registry(self, tmp_path: Path) -> ModelRegistry:
        return ModelRegistry(path=tmp_path / "registry.json")

    def test_empty_registry(self, registry: ModelRegistry) -> None:
        assert registry.all() == []
        assert registry.champion() is None
        assert registry.table() == "(registry is empty)"

    def test_register_and_read_back(self, registry: ModelRegistry) -> None:
        registry.register(card("v1"))
        assert registry.get("v1") is not None
        assert registry.get("v1").stage is Stage.CANDIDATE
        assert registry.get("missing") is None

    def test_re_registering_replaces(self, registry: ModelRegistry) -> None:
        registry.register(card("v1", f1=0.80))
        registry.register(card("v1", f1=0.90))
        assert len(registry.all()) == 1
        assert registry.get("v1").metric("weighted_f1") == 0.90

    def test_promotion_makes_a_champion(self, registry: ModelRegistry) -> None:
        registry.register(card("v1"))
        decision = registry.promote("v1")

        assert decision.promote
        assert registry.champion().model_id == "v1"

    def test_promotion_archives_the_previous_champion(self, registry: ModelRegistry) -> None:
        registry.register(card("v1", f1=0.90))
        registry.promote("v1")
        registry.register(card("v2", f1=0.94))
        registry.promote("v2")

        assert registry.champion().model_id == "v2"
        assert registry.get("v1").stage is Stage.ARCHIVED
        assert any("superseded by v2" in note for note in registry.get("v1").notes)

    def test_a_held_candidate_does_not_become_champion(self, registry: ModelRegistry) -> None:
        registry.register(card("v1", f1=0.90))
        registry.promote("v1")
        registry.register(card("v2", f1=0.85))

        decision = registry.promote("v2")
        assert not decision.promote
        assert registry.champion().model_id == "v1"
        assert registry.get("v2").stage is Stage.CANDIDATE

    def test_force_promotes_over_a_failed_gate_and_records_it(
        self, registry: ModelRegistry
    ) -> None:
        registry.register(card("v1", f1=0.90))
        registry.promote("v1")
        registry.register(card("v2", f1=0.85))

        decision = registry.promote("v2", force=True)
        assert decision.promote
        assert registry.champion().model_id == "v2"
        assert any("force-promoted" in note for note in registry.get("v2").notes)

    def test_rollback_restores_the_previous_champion(self, registry: ModelRegistry) -> None:
        registry.register(card("v1", f1=0.90))
        registry.promote("v1")
        registry.register(card("v2", f1=0.94))
        registry.promote("v2")

        restored = registry.rollback()
        assert restored.model_id == "v1"
        assert registry.champion().model_id == "v1"
        assert registry.get("v2").stage is Stage.ARCHIVED

    def test_rollback_with_nothing_archived_is_a_no_op(self, registry: ModelRegistry) -> None:
        registry.register(card("v1"))
        registry.promote("v1")
        assert registry.rollback() is None

    def test_promoting_an_unknown_model_raises(self, registry: ModelRegistry) -> None:
        with pytest.raises(KeyError):
            registry.promote("nope")

    def test_by_stage(self, registry: ModelRegistry) -> None:
        registry.register(card("v1"))
        registry.promote("v1")
        registry.register(card("v2", f1=0.80))

        assert [c.model_id for c in registry.by_stage(Stage.CHAMPION)] == ["v1"]
        assert [c.model_id for c in registry.by_stage(Stage.CANDIDATE)] == ["v2"]

    def test_table_renders(self, registry: ModelRegistry) -> None:
        registry.register(card("v1"))
        rendered = registry.table()
        assert "v1" in rendered and "weighted" not in rendered.lower()[:20]
        assert "F1" in rendered

    def test_card_round_trip(self, registry: ModelRegistry) -> None:
        original = card("v1")
        registry.register(original)
        restored = registry.get("v1")

        assert restored.to_dict()["metrics"] == original.to_dict()["metrics"]
        assert restored.per_key_f1 == original.per_key_f1
        assert restored.corpus_fingerprint == original.corpus_fingerprint

    def test_config_hash_is_stable_and_discriminating(self) -> None:
        a = card("v1")
        a.training_config = {"r": 16, "lr": 1e-4}
        b = card("v2")
        b.training_config = {"lr": 1e-4, "r": 16}
        c = card("v3")
        c.training_config = {"r": 32, "lr": 1e-4}

        assert a.config_hash() == b.config_hash()
        assert a.config_hash() != c.config_hash()


class TestPipelinePlan:
    @pytest.fixture
    def config(self) -> PipelineConfig:
        return PipelineConfig(
            role_arn="arn:aws:iam::000000000000:role/Throughline",
            bucket="throughline-artifacts",
            schema="invoice",
        )

    def test_plan_has_every_stage(self, config: PipelineConfig) -> None:
        names = [step.name for step in build_plan(config)]
        assert names == [
            "BuildDataset", "TrainLoRA", "Evaluate",
            "PromotionGate", "RegisterModel", "DeployEndpoint", "GateFailed",
        ]

    def test_plan_is_valid(self, config: PipelineConfig) -> None:
        assert validate_plan(config) == []

    def test_deployment_is_downstream_of_the_gate(self, config: PipelineConfig) -> None:
        """A pipeline that always deploys what it trained ships regressions on a schedule."""
        plan = {step.name: step for step in build_plan(config)}
        assert "PromotionGate" in plan["RegisterModel"].depends_on
        assert "RegisterModel" in plan["DeployEndpoint"].depends_on
        assert plan["PromotionGate"].kind == "condition"

    def test_the_failure_branch_exists(self, config: PipelineConfig) -> None:
        plan = {step.name: step for step in build_plan(config)}
        assert plan["GateFailed"].kind == "fail"
        assert "PromotionGate" in plan["GateFailed"].depends_on

    def test_gate_thresholds_are_recorded_in_the_step(self, config: PipelineConfig) -> None:
        step = next(s for s in build_plan(config) if s.name == "PromotionGate")
        assert str(config.gate.min_improvement) in step.description
        assert str(config.gate.per_key_tolerance) in step.description

    def test_s3_uris_are_scoped_to_the_schema(self, config: PipelineConfig) -> None:
        step = next(s for s in build_plan(config) if s.name == "BuildDataset")
        assert step.inputs["corpus"].endswith("/corpus/invoice")
        assert step.inputs["corpus"].startswith("s3://throughline-artifacts/throughline/")

    def test_s3_requires_a_bucket(self) -> None:
        with pytest.raises(ValueError, match="bucket"):
            PipelineConfig().s3("corpus")

    def test_render_is_readable(self, config: PipelineConfig) -> None:
        rendered = render_plan(config)
        assert "TrainLoRA [training]" in rendered
        assert "PromotionGate [condition]" in rendered

    def test_export_writes_a_diffable_definition(
        self, config: PipelineConfig, tmp_path: Path
    ) -> None:
        import json

        path = export_definition(config, tmp_path / "pipeline.json")
        payload = json.loads(Path(path).read_text())

        assert payload["problems"] == []
        assert len(payload["steps"]) == 7
        assert payload["config"]["schema"] == "invoice"

    def test_building_without_credentials_config_raises(self) -> None:
        from throughline.training.pipeline import build_pipeline

        with pytest.raises((ValueError, RuntimeError)):
            build_pipeline(PipelineConfig(bucket="b"))  # no role_arn
