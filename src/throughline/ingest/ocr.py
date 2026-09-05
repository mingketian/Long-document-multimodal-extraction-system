"""OCR / layout providers.

The pipeline depends on the :class:`OcrProvider` protocol, never on a specific
engine. Three implementations ship here:

* :class:`TextractProvider` - Amazon Textract ``AnalyzeDocument`` with LAYOUT and
  TABLES, which is what the AWS GenAI IDP accelerator already has wired up.
* :class:`PyMuPdfProvider` - the digital-text path. Most enterprise PDFs carry a
  text layer; reading it is faster, free, and more accurate than re-OCRing pixels.
* :class:`JsonFixtureProvider` - reads pre-extracted layout from disk, so tests and
  the offline demo run with no cloud calls and no model weights.

Results pass through the content-addressed cache, which is what makes re-running a
document after a prompt change cost nothing in OCR.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from throughline.ingest.layout import (
    BlockType,
    BoundingBox,
    Document,
    LayoutBlock,
    Page,
    assign_reading_order,
    infer_block_type,
    normalise_bbox,
)

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class OcrProvider(Protocol):
    """Anything that can turn a source file into a :class:`Document`."""

    name: str

    def extract(self, source: str | Path, *, document_id: str | None = None) -> Document:
        """Read ``source`` and return a fully populated document."""
        ...


def _document_id_for(source: str | Path) -> str:
    return Path(source).stem


# ── digital-text PDFs ─────────────────────────────────────────────────────
@dataclass
class PyMuPdfProvider:
    """Read the embedded text layer of a PDF with PyMuPDF.

    Falls back to raising :class:`RuntimeError` when a page has no text layer, so the
    caller can route that document to a real OCR engine rather than silently
    extracting from an empty page.
    """

    name: str = "pymupdf"
    render_dpi: int = 200
    image_dir: str | None = None
    min_chars_per_page: int = 20

    def extract(self, source: str | Path, *, document_id: str | None = None) -> Document:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError(
                "PyMuPdfProvider needs pymupdf. Install with: pip install 'throughline[ingest]'"
            ) from exc

        source = Path(source)
        document = Document(
            document_id=document_id or _document_id_for(source),
            source_path=str(source),
            metadata={"provider": self.name},
        )

        image_dir = Path(self.image_dir) if self.image_dir else None
        if image_dir:
            image_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(source) as pdf:
            for index, raw_page in enumerate(pdf, start=1):
                rect = raw_page.rect
                page = Page(
                    page_number=index,
                    width=int(rect.width),
                    height=int(rect.height),
                )

                if image_dir:
                    matrix = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)
                    image_path = image_dir / f"{document.document_id}_p{index:04d}.png"
                    raw_page.get_pixmap(matrix=matrix).save(image_path)
                    page.image_path = str(image_path)

                blocks: list[LayoutBlock] = []
                for block_index, raw_block in enumerate(raw_page.get_text("blocks")):
                    x0, y0, x1, y1, text = raw_block[:5]
                    text = str(text).strip()
                    if not text:
                        continue
                    bbox = normalise_bbox(
                        x0, y0, x1, y1, width=rect.width, height=rect.height
                    )
                    blocks.append(
                        LayoutBlock(
                            block_id=f"p{index}b{block_index}",
                            page_number=index,
                            block_type=infer_block_type(text, bbox),
                            text=text,
                            bbox=bbox,
                        )
                    )

                if sum(len(b.text) for b in blocks) < self.min_chars_per_page:
                    raise RuntimeError(
                        f"Page {index} of {source} has no usable text layer "
                        f"({sum(len(b.text) for b in blocks)} chars). Route this document "
                        f"to an OCR provider instead."
                    )

                page.blocks = assign_reading_order(blocks)
                document.pages.append(page)

        return document


# ── Amazon Textract ───────────────────────────────────────────────────────
_TEXTRACT_BLOCK_TYPES = {
    "LAYOUT_TITLE": BlockType.TITLE,
    "LAYOUT_SECTION_HEADER": BlockType.HEADING,
    "LAYOUT_HEADER": BlockType.HEADER,
    "LAYOUT_FOOTER": BlockType.FOOTER,
    "LAYOUT_TEXT": BlockType.PARAGRAPH,
    "LAYOUT_TABLE": BlockType.TABLE,
    "LAYOUT_KEY_VALUE": BlockType.KEY_VALUE,
    "LAYOUT_FIGURE": BlockType.FIGURE,
    "TABLE": BlockType.TABLE,
    "CELL": BlockType.TABLE_ROW,
}


@dataclass
class TextractProvider:
    """Amazon Textract ``AnalyzeDocument`` with the LAYOUT and TABLES feature types.

    Args:
        bucket: S3 bucket holding the source document. Textract reads from S3 for
            multi-page documents; single-page calls may pass bytes instead.
        region: AWS region for the Textract client.
        feature_types: Textract features to request.
        max_pages: Safety bound so a runaway document cannot exhaust the budget.
    """

    bucket: str | None = None
    region: str = "us-east-1"
    feature_types: tuple[str, ...] = ("LAYOUT", "TABLES", "FORMS")
    max_pages: int = 500
    name: str = "textract"

    def _client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError(
                "TextractProvider needs boto3. Install with: pip install 'throughline[aws]'"
            ) from exc
        return boto3.client("textract", region_name=self.region)

    def extract(self, source: str | Path, *, document_id: str | None = None) -> Document:
        client = self._client()
        source = str(source)

        if source.startswith("s3://"):
            _, _, remainder = source.partition("s3://")
            bucket, _, key = remainder.partition("/")
        elif self.bucket:
            bucket, key = self.bucket, source
        else:
            raise ValueError(
                "TextractProvider needs either an s3:// source or a configured bucket."
            )

        started = client.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
            FeatureTypes=list(self.feature_types),
        )
        blocks = self._collect_blocks(client, started["JobId"])
        return self._to_document(blocks, document_id or Path(key).stem, source)

    def _collect_blocks(self, client: Any, job_id: str) -> list[dict[str, Any]]:
        import time

        blocks: list[dict[str, Any]] = []
        next_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {"JobId": job_id}
            if next_token:
                kwargs["NextToken"] = next_token
            response = client.get_document_analysis(**kwargs)
            status = response["JobStatus"]

            if status == "IN_PROGRESS":
                time.sleep(2)
                continue
            if status != "SUCCEEDED":
                raise RuntimeError(f"Textract job {job_id} finished with status {status}.")

            blocks.extend(response.get("Blocks", []))
            next_token = response.get("NextToken")
            if not next_token:
                return blocks

    def _to_document(
        self, raw_blocks: list[dict[str, Any]], document_id: str, source: str
    ) -> Document:
        document = Document(
            document_id=document_id, source_path=source, metadata={"provider": self.name}
        )
        by_id = {block["Id"]: block for block in raw_blocks}
        by_page: dict[int, list[LayoutBlock]] = {}

        def block_text(block: dict[str, Any]) -> str:
            if "Text" in block:
                return str(block["Text"])
            words: list[str] = []
            for relationship in block.get("Relationships", []):
                if relationship["Type"] != "CHILD":
                    continue
                for child_id in relationship["Ids"]:
                    child = by_id.get(child_id, {})
                    if child.get("BlockType") in {"WORD", "LINE"}:
                        words.append(str(child.get("Text", "")))
            return " ".join(word for word in words if word)

        for index, block in enumerate(raw_blocks):
            block_type = block.get("BlockType", "")
            if block_type not in _TEXTRACT_BLOCK_TYPES:
                continue
            geometry = block.get("Geometry", {}).get("BoundingBox")
            if not geometry:
                continue
            text = block_text(block).strip()
            if not text:
                continue

            page_number = int(block.get("Page", 1))
            if page_number > self.max_pages:
                LOGGER.warning("Truncating document at page %s (max_pages).", self.max_pages)
                break

            bbox = BoundingBox(
                x0=float(geometry["Left"]),
                y0=float(geometry["Top"]),
                x1=min(float(geometry["Left"]) + float(geometry["Width"]), 1.0),
                y1=min(float(geometry["Top"]) + float(geometry["Height"]), 1.0),
            )
            by_page.setdefault(page_number, []).append(
                LayoutBlock(
                    block_id=block.get("Id", f"p{page_number}b{index}"),
                    page_number=page_number,
                    block_type=_TEXTRACT_BLOCK_TYPES[block_type],
                    text=text,
                    bbox=bbox,
                    confidence=float(block.get("Confidence", 100.0)) / 100.0,
                    row_index=block.get("RowIndex"),
                )
            )

        for page_number in sorted(by_page):
            document.pages.append(
                Page(page_number=page_number, blocks=assign_reading_order(by_page[page_number]))
            )
        return document


# ── fixtures ──────────────────────────────────────────────────────────────
@dataclass
class JsonFixtureProvider:
    """Load a pre-extracted :class:`Document` from JSON.

    This is what makes the pipeline testable and the demo runnable with no cloud
    credentials, no GPU, and no model weights.
    """

    name: str = "json-fixture"

    def extract(self, source: str | Path, *, document_id: str | None = None) -> Document:
        document = Document.from_json_file(source)
        if document_id:
            document.document_id = document_id
        for page in document.pages:
            page.blocks = assign_reading_order(page.blocks)
        return document


# ── caching wrapper ───────────────────────────────────────────────────────
@dataclass
class CachedOcrProvider:
    """Wrap any provider with a content-addressed on-disk cache.

    Keyed on the source file's bytes plus the wrapped provider's name, so switching
    OCR engines or editing the source invalidates the entry, while re-running the
    same document after a prompt or schema change is free. This is one of the three
    levers behind the productionised pipeline's latency reduction.
    """

    provider: OcrProvider
    cache_dir: str | Path = ".cache/ocr"
    enabled: bool = True

    @property
    def name(self) -> str:
        return f"cached:{self.provider.name}"

    def _key(self, source: str | Path) -> str:
        digest = hashlib.sha256()
        digest.update(self.provider.name.encode("utf-8"))
        path = Path(source)
        if path.exists():
            digest.update(path.read_bytes())
        else:  # remote sources are keyed by URI
            digest.update(str(source).encode("utf-8"))
        return digest.hexdigest()[:32]

    def extract(self, source: str | Path, *, document_id: str | None = None) -> Document:
        if not self.enabled:
            return self.provider.extract(source, document_id=document_id)

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self._key(source)}.json"

        if cache_path.exists():
            LOGGER.debug("OCR cache hit for %s", source)
            document = Document.from_json_file(cache_path)
            if document_id:
                document.document_id = document_id
            return document

        document = self.provider.extract(source, document_id=document_id)
        cache_path.write_text(
            json.dumps(document.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.debug("OCR cache store for %s", source)
        return document
