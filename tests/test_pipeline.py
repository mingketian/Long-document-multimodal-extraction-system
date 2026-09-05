"""End-to-end pipeline behaviour: attribution, early exit, caching, retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from throughline.attribution.evidence import AttributionStatus, attribute
from throughline.caching.store import CachedBackend, PromptCache
from throughline.grouping.page_groups import GroupingConfig, partition
from throughline.ingest.ocr import JsonFixtureProvider
from throughline.models.base import GenerationConfig, GenerationResult
from throughline.models.rule_based import EchoBackend, RuleBasedBackend
from throughline.pipeline.early_exit import EarlyExitPolicy, ExitReason
from throughline.pipeline.orchestrator import ExtractionPipeline, PipelineConfig
from throughline.prompting.templates import build_prompt
from throughline.retrieval.relevant_pages import RelevantPageRetriever
from throughline.schema import registry
from throughline.state.cross_page import CrossPageState

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
INVOICE = registry.get("invoice")
AGREEMENT = registry.get("service_agreement")


@pytest.fixture(scope="module")
def invoice_document():
    return JsonFixtureProvider().extract(EXAMPLES / "documents" / "invoice_0001.json")


@pytest.fixture(scope="module")
def agreement_document():
    return JsonFixtureProvider().extract(EXAMPLES / "documents" / "agreement_0001.json")


@pytest.fixture(scope="module")
def invoice_gold():
    payload = json.loads((EXAMPLES / "corpus" / "invoice_0001.json").read_text())
    return payload["gold"]


class TestEndToEnd:
    def test_extraction_produces_a_valid_record(self, invoice_document) -> None:
        result = ExtractionPipeline(RuleBasedBackend(), INVOICE).run(invoice_document)

        assert result.is_valid, result.validation.summary()
        assert result.record["invoice_number"].startswith("INV-")
        assert result.groups_processed >= 1

    def test_every_page_is_read_without_early_exit(self, invoice_document) -> None:
        pipeline = ExtractionPipeline(
            RuleBasedBackend(),
            INVOICE,
            PipelineConfig(early_exit=EarlyExitPolicy(enabled=False)),
        )
        result = pipeline.run(invoice_document)
        assert result.pages_read == invoice_document.page_count

    def test_table_rows_are_recovered_without_overlap_duplicates(
        self, invoice_document, invoice_gold
    ) -> None:
        """The overlap re-shows boundary rows; the row key must collapse them."""
        pipeline = ExtractionPipeline(
            RuleBasedBackend(),
            INVOICE,
            PipelineConfig(early_exit=EarlyExitPolicy(enabled=False)),
        )
        result = pipeline.run(invoice_document)
        assert len(result.record["line_items"]) == len(invoice_gold["line_items"])

    def test_totals_come_from_the_last_page_not_the_first(
        self, invoice_document, invoice_gold
    ) -> None:
        result = ExtractionPipeline(RuleBasedBackend(), INVOICE).run(invoice_document)
        assert result.record["total_amount"] == invoice_gold["total_amount"]
        assert result.record["total_amount"] != result.record.get("subtotal")

    def test_result_serialises(self, invoice_document) -> None:
        result = ExtractionPipeline(RuleBasedBackend(), INVOICE).run(invoice_document)
        payload = json.loads(json.dumps(result.to_dict(), default=str))
        assert payload["document_id"] == invoice_document.document_id
        assert payload["exit"]["groups_total"] >= 1

    def test_batch_isolates_a_failing_document(self, invoice_document) -> None:
        class Exploding:
            name = "boom"

            def generate(self, prompt, config=None):
                raise RuntimeError("backend is down")

        pipeline = ExtractionPipeline(Exploding(), INVOICE)
        results = pipeline.run_batch([invoice_document])

        assert len(results) == 1
        assert results[0].record == {}
        assert not results[0].is_valid


class TestEarlyExit:
    def test_agreement_stops_before_reading_every_page(self, agreement_document) -> None:
        pipeline = ExtractionPipeline(
            RuleBasedBackend(),
            AGREEMENT,
            PipelineConfig(
                early_exit=EarlyExitPolicy(enabled=True, min_groups=1, require_evidence=False)
            ),
        )
        result = pipeline.run(agreement_document)

        assert result.groups_processed < result.groups_total
        assert result.exit_reason is not None and result.exit_reason.is_early

    def test_disabled_policy_reads_everything(self, agreement_document) -> None:
        pipeline = ExtractionPipeline(
            RuleBasedBackend(),
            AGREEMENT,
            PipelineConfig(early_exit=EarlyExitPolicy(enabled=False)),
        )
        result = pipeline.run(agreement_document)
        assert result.groups_processed == result.groups_total
        assert result.exit_reason is ExitReason.ALL_GROUPS_PROCESSED

    def test_an_open_table_blocks_stopping(self) -> None:
        """Stopping mid-table is the failure mode this guard exists to prevent."""
        policy = EarlyExitPolicy(enabled=True, min_groups=1, respect_open_tables=True)
        state = CrossPageState(schema=INVOICE)
        for name, value in (
            ("invoice_number", "INV-1"),
            ("invoice_date", "2026-01-01"),
            ("vendor_name", "Acme"),
            ("total_amount", "$1.00"),
        ):
            state.update_field(name, value, group_index=0, confidence=1.0)
        state.append_rows("line_items", [{"line_number": 1, "description": "A"}], group_index=0)
        state.mark_table_open("line_items")

        decision = policy.evaluate(state, INVOICE, groups_processed=1, groups_total=9)
        assert not decision.should_stop
        assert "still open" in decision.detail

    def test_max_groups_is_a_hard_bound(self, agreement_document) -> None:
        pipeline = ExtractionPipeline(
            RuleBasedBackend(),
            AGREEMENT,
            PipelineConfig(early_exit=EarlyExitPolicy(enabled=True, max_groups=1)),
        )
        result = pipeline.run(agreement_document)
        assert result.groups_processed == 1
        assert result.exit_reason is ExitReason.BUDGET_EXHAUSTED

    def test_unproductive_groups_exhaust_patience(self) -> None:
        policy = EarlyExitPolicy(enabled=True, min_groups=1, patience=2)
        state = CrossPageState(schema=INVOICE)

        policy.note_group(changed=False)
        policy.note_group(changed=False)
        decision = policy.evaluate(state, INVOICE, groups_processed=3, groups_total=9)

        assert decision.should_stop
        assert decision.reason is ExitReason.NO_NEW_INFORMATION


class TestAttribution:
    def test_block_id_resolves_exactly(self, invoice_document) -> None:
        pages = invoice_document.pages[:1]
        block = pages[0].blocks[0]
        result = attribute({"block_id": block.block_id, "confidence": 0.9}, block.text, pages)

        assert result.status is AttributionStatus.BLOCK_ID
        assert result.ref.page_number == 1

    def test_bad_block_id_falls_back_to_the_quote(self, invoice_document) -> None:
        pages = invoice_document.pages[:1]
        block = pages[0].blocks[2]
        result = attribute({"block_id": "hallucinated", "quote": block.text}, "x", pages)

        assert result.status is AttributionStatus.QUOTE
        assert result.ref.block_id == block.block_id

    def test_value_search_is_the_last_resort(self, invoice_document) -> None:
        pages = invoice_document.pages[:1]
        value = pages[0].blocks[2].text.split(": ")[-1]
        result = attribute({"block_id": None, "quote": ""}, value, pages)

        assert result.status is AttributionStatus.VALUE

    def test_unresolvable_citation_is_reported_not_accepted(self, invoice_document) -> None:
        result = attribute(
            {"block_id": "nope", "quote": "text that appears nowhere at all"},
            "also absent",
            invoice_document.pages[:1],
        )
        assert result.status is AttributionStatus.UNVERIFIED
        assert result.ref is None

    def test_pipeline_reports_citation_precision(self, invoice_document) -> None:
        result = ExtractionPipeline(RuleBasedBackend(), INVOICE).run(invoice_document)
        assert result.attribution.total_claims > 0
        assert 0.0 <= result.attribution.citation_precision <= 1.0


class TestCaching:
    def test_second_run_hits_the_cache(self, invoice_document, tmp_path: Path) -> None:
        cache = PromptCache(cache_dir=tmp_path / "prompts")
        backend = CachedBackend(RuleBasedBackend(), cache)
        config = PipelineConfig(early_exit=EarlyExitPolicy(enabled=False))

        ExtractionPipeline(backend, INVOICE, config).run(invoice_document)
        assert cache.stats.hits == 0
        assert cache.stats.stores > 0

        ExtractionPipeline(backend, INVOICE, config).run(invoice_document)
        assert cache.stats.hits > 0

    def test_cached_and_uncached_records_agree(self, invoice_document, tmp_path: Path) -> None:
        cache = PromptCache(cache_dir=tmp_path / "prompts")
        config = PipelineConfig(early_exit=EarlyExitPolicy(enabled=False))

        cold = ExtractionPipeline(RuleBasedBackend(), INVOICE, config).run(invoice_document)
        warm_backend = CachedBackend(RuleBasedBackend(), cache)
        ExtractionPipeline(warm_backend, INVOICE, config).run(invoice_document)
        warm = ExtractionPipeline(warm_backend, INVOICE, config).run(invoice_document)

        assert cold.record == warm.record

    def test_clear_empties_the_cache(self, tmp_path: Path) -> None:
        cache = PromptCache(cache_dir=tmp_path / "prompts")
        cache.put("abc123", GenerationResult(text="{}", backend="test"))
        assert cache.size()[0] == 1
        assert cache.clear() == 1
        assert cache.size()[0] == 0

    def test_config_change_invalidates_the_key(self, invoice_document) -> None:
        from throughline.caching.store import prompt_cache_key

        group = partition(invoice_document, GroupingConfig())[0]
        prompt = build_prompt(INVOICE, group)

        greedy = prompt_cache_key(prompt, GenerationConfig(), "b")
        sampled = prompt_cache_key(prompt, GenerationConfig(do_sample=True), "b")
        assert greedy != sampled


class TestRetrieval:
    def test_header_fields_rank_the_first_page_highest(self, invoice_document) -> None:
        retriever = RelevantPageRetriever(invoice_document, INVOICE)
        assert retriever.top_pages(["invoice_number"], k=1) == [1]

    def test_totals_rank_the_last_page_highest(self, invoice_document) -> None:
        retriever = RelevantPageRetriever(invoice_document, INVOICE)
        top = retriever.top_pages(["total_amount"], k=1)[0]
        assert top == invoice_document.page_count

    def test_buried_clause_is_found_in_a_long_agreement(self, agreement_document) -> None:
        retriever = RelevantPageRetriever(agreement_document, AGREEMENT)
        ranked = retriever.top_pages(["governing_law"], k=3)
        assert ranked[0] >= agreement_document.page_count - 4

    def test_group_ranking_returns_every_group(self, agreement_document) -> None:
        groups = partition(agreement_document, GroupingConfig())
        retriever = RelevantPageRetriever(agreement_document, AGREEMENT)
        ranked = retriever.rank_groups(groups, ["governing_law"])
        assert len(ranked) == len(groups)


class TestErrorHandling:
    def test_unparseable_output_is_recorded_not_raised(self, invoice_document) -> None:
        pipeline = ExtractionPipeline(
            EchoBackend(response="I am terribly sorry, I cannot help with that."),
            INVOICE,
            PipelineConfig(repair_attempts=0, early_exit=EarlyExitPolicy(enabled=False)),
        )
        result = pipeline.run(invoice_document)

        assert result.record == {}
        assert any(trace.error for trace in result.traces)

    def test_fail_fast_reraises(self, invoice_document) -> None:
        class Exploding:
            name = "boom"

            def generate(self, prompt, config=None):
                raise RuntimeError("backend is down")

        pipeline = ExtractionPipeline(Exploding(), INVOICE, PipelineConfig(fail_fast=True))
        with pytest.raises(RuntimeError, match="backend is down"):
            pipeline.run(invoice_document)

    def test_empty_envelope_yields_an_invalid_record(self, invoice_document) -> None:
        pipeline = ExtractionPipeline(
            EchoBackend(response="{}"),
            INVOICE,
            PipelineConfig(early_exit=EarlyExitPolicy(enabled=False)),
        )
        result = pipeline.run(invoice_document)
        assert not result.is_valid
