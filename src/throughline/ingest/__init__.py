"""Document ingestion: pages, layout blocks, and the OCR providers that build them."""

from throughline.ingest.layout import (
    BlockType,
    BoundingBox,
    Document,
    LayoutBlock,
    Page,
    assign_reading_order,
)
from throughline.ingest.ocr import (
    CachedOcrProvider,
    JsonFixtureProvider,
    OcrProvider,
    PyMuPdfProvider,
    TextractProvider,
)

__all__ = [
    "BlockType",
    "BoundingBox",
    "CachedOcrProvider",
    "Document",
    "JsonFixtureProvider",
    "LayoutBlock",
    "OcrProvider",
    "Page",
    "PyMuPdfProvider",
    "TextractProvider",
    "assign_reading_order",
]
