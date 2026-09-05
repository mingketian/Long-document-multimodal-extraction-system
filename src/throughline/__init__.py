"""Throughline - cross-page state for long-document multimodal extraction.

A schema-constrained extraction pipeline for documents too long to fit in one
context window. Pages are partitioned into bounded, overlapping groups; a cross-page
state carries extracted fields and evidence references from one group to the next;
every value is attributed back to the block it was read from; and the run stops as
soon as the schema is satisfied and evidenced.

Quick start::

    from throughline import ExtractionPipeline, PipelineConfig
    from throughline.ingest import JsonFixtureProvider
    from throughline.models import RuleBasedBackend
    from throughline.schema import registry

    document = JsonFixtureProvider().extract("examples/documents/invoice_0001.json")
    pipeline = ExtractionPipeline(RuleBasedBackend(), registry.get("invoice"))
    result = pipeline.run(document)

    print(result.summary())
    print(result.record)

Or from a named profile::

    from throughline.config import profile

    result = profile("balanced").build_pipeline().run(document)
"""

from __future__ import annotations

__version__ = "0.3.0"

from throughline.grouping.page_groups import GroupingConfig, PageGroup, partition
from throughline.ingest.layout import BlockType, BoundingBox, Document, LayoutBlock, Page
from throughline.models.base import GenerationConfig, GenerationResult
from throughline.pipeline.early_exit import EarlyExitPolicy, ExitReason
from throughline.pipeline.orchestrator import (
    ExtractionPipeline,
    ExtractionResult,
    PipelineConfig,
)
from throughline.schema.spec import (
    Cardinality,
    ExtractionSchema,
    FieldSpec,
    FieldType,
    TableSpec,
)
from throughline.schema.validate import ValidationReport, validate_record
from throughline.state.cross_page import CrossPageState, EvidenceRef

__all__ = [
    "BlockType",
    "BoundingBox",
    "Cardinality",
    "CrossPageState",
    "Document",
    "EarlyExitPolicy",
    "EvidenceRef",
    "ExitReason",
    "ExtractionPipeline",
    "ExtractionResult",
    "ExtractionSchema",
    "FieldSpec",
    "FieldType",
    "GenerationConfig",
    "GenerationResult",
    "GroupingConfig",
    "LayoutBlock",
    "Page",
    "PageGroup",
    "PipelineConfig",
    "TableSpec",
    "ValidationReport",
    "__version__",
    "partition",
    "validate_record",
]
