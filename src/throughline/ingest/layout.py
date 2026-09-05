"""Document, page and layout primitives.

The pipeline never passes raw pixels around on its own. Every page carries both its
rendered image *and* the OCR/layout signal extracted from it, because the two are
complementary: the VLM reads the image, while the layout blocks give the system
exact character spans and bounding boxes to attribute evidence to. Attribution is
only possible because the text the model quotes can be matched back to a block that
knows its own page number and coordinates.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BlockType(str, Enum):
    """Coarse layout roles, as produced by most OCR/layout engines."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    KEY_VALUE = "key_value"
    HEADER = "page_header"
    FOOTER = "page_footer"
    FIGURE = "figure"
    SIGNATURE = "signature"
    OTHER = "other"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in normalised page coordinates (0-1, origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x0 <= self.x1 <= 1.0 and 0.0 <= self.y0 <= self.y1 <= 1.0):
            raise ValueError(f"Invalid normalised bounding box: {self}")

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union, used to match predicted to gold evidence."""
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        intersection = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values: Iterable[float]) -> BoundingBox:
        x0, y0, x1, y1 = values
        return cls(float(x0), float(y0), float(x1), float(y1))


@dataclass(frozen=True)
class LayoutBlock:
    """One OCR/layout unit on a page."""

    block_id: str
    page_number: int
    block_type: BlockType
    text: str
    bbox: BoundingBox
    confidence: float = 1.0
    reading_order: int = 0
    row_index: int | None = None
    """Set for TABLE_ROW blocks; the row's index within its printed table."""

    def render(self) -> str:
        """The form shown to the model in the prompt: role-tagged and addressable."""
        return f"[{self.block_id}|{self.block_type.value}] {self.text}"


@dataclass
class Page:
    """A single page: its rendered image, its layout blocks, and derived text."""

    page_number: int
    image_path: str | None = None
    blocks: list[LayoutBlock] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def text(self) -> str:
        """Full page text in reading order."""
        ordered = sorted(self.blocks, key=lambda b: (b.reading_order, b.bbox.y0, b.bbox.x0))
        return "\n".join(block.text for block in ordered if block.text.strip())

    def render_layout(self, *, max_chars: int = 6_000) -> str:
        """Role-tagged layout text for prompt assembly, truncated at a hard budget."""
        ordered = sorted(self.blocks, key=lambda b: (b.reading_order, b.bbox.y0, b.bbox.x0))
        lines: list[str] = []
        used = 0
        for block in ordered:
            if not block.text.strip():
                continue
            rendered = block.render()
            if used + len(rendered) > max_chars:
                lines.append("[...truncated]")
                break
            lines.append(rendered)
            used += len(rendered) + 1
        return "\n".join(lines)

    def blocks_of_type(self, *types: BlockType) -> list[LayoutBlock]:
        wanted = set(types)
        return [block for block in self.blocks if block.block_type in wanted]

    def find_block(self, block_id: str) -> LayoutBlock | None:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "blocks": [
                {
                    "block_id": b.block_id,
                    "block_type": b.block_type.value,
                    "text": b.text,
                    "bbox": b.bbox.to_list(),
                    "confidence": b.confidence,
                    "reading_order": b.reading_order,
                    "row_index": b.row_index,
                }
                for b in self.blocks
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Page:
        page = cls(
            page_number=int(payload["page_number"]),
            image_path=payload.get("image_path"),
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
        )
        for raw in payload.get("blocks", []):
            page.blocks.append(
                LayoutBlock(
                    block_id=raw["block_id"],
                    page_number=page.page_number,
                    block_type=BlockType(raw.get("block_type", "other")),
                    text=raw.get("text", ""),
                    bbox=BoundingBox.from_list(raw["bbox"]),
                    confidence=float(raw.get("confidence", 1.0)),
                    reading_order=int(raw.get("reading_order", 0)),
                    row_index=raw.get("row_index"),
                )
            )
        return page


@dataclass
class Document:
    """A long document: an ordered list of pages plus identifying metadata."""

    document_id: str
    pages: list[Page] = field(default_factory=list)
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pages)

    def __iter__(self) -> Iterator[Page]:
        return iter(self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, page_number: int) -> Page | None:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def find_block(self, block_id: str) -> LayoutBlock | None:
        """Resolve a block id anywhere in the document - the attribution entry point."""
        for page in self.pages:
            block = page.find_block(block_id)
            if block is not None:
                return block
        return None

    def content_hash(self) -> str:
        """Stable hash over page text, used as the OCR/prompt cache key."""
        digest = hashlib.sha256()
        digest.update(self.document_id.encode("utf-8"))
        for page in self.pages:
            digest.update(f"\x00{page.page_number}\x00".encode())
            digest.update(page.text.encode("utf-8"))
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "metadata": self.metadata,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Document:
        return cls(
            document_id=payload["document_id"],
            source_path=payload.get("source_path"),
            metadata=payload.get("metadata", {}),
            pages=[Page.from_dict(raw) for raw in payload.get("pages", [])],
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> Document:
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json_file(self, path: str | Path) -> None:
        import json

        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


# ── reading order ─────────────────────────────────────────────────────────
_HEADER_BAND = 0.08
_FOOTER_BAND = 0.92


def assign_reading_order(blocks: list[LayoutBlock], *, column_tolerance: float = 0.05) -> list[LayoutBlock]:
    """Sort blocks into human reading order, handling simple two-column layouts.

    Blocks whose horizontal centres cluster on the left or right half are treated as
    separate columns and read left column fully, then right - which is what a person
    does, and what makes quoted evidence line up with the model's own reading path.
    """
    if not blocks:
        return []

    centres = [block.bbox.centre[0] for block in blocks]
    left = [c for c in centres if c < 0.5]
    right = [c for c in centres if c >= 0.5]

    two_column = (
        len(left) >= 3
        and len(right) >= 3
        and (min(right) - max(left)) > column_tolerance
    )

    def sort_key(block: LayoutBlock) -> tuple[int, float, float]:
        column = 0
        if two_column and block.bbox.centre[0] >= 0.5:
            column = 1
        return (column, round(block.bbox.y0, 3), block.bbox.x0)

    ordered = sorted(blocks, key=sort_key)
    return [
        LayoutBlock(
            block_id=block.block_id,
            page_number=block.page_number,
            block_type=block.block_type,
            text=block.text,
            bbox=block.bbox,
            confidence=block.confidence,
            reading_order=index,
            row_index=block.row_index,
        )
        for index, block in enumerate(ordered)
    ]


def infer_block_type(text: str, bbox: BoundingBox, *, default: BlockType = BlockType.PARAGRAPH) -> BlockType:
    """Heuristic role assignment for OCR engines that do not classify blocks."""
    stripped = text.strip()
    if not stripped:
        return BlockType.OTHER
    if bbox.y1 <= _HEADER_BAND:
        return BlockType.HEADER
    if bbox.y0 >= _FOOTER_BAND:
        return BlockType.FOOTER
    if re.match(r"^[A-Z0-9][A-Z0-9 \-&/,.]{3,}$", stripped) and len(stripped) < 80:
        return BlockType.HEADING
    if re.match(r"^\s*\S[^:\n]{0,40}:\s*\S", stripped) and "\n" not in stripped:
        return BlockType.KEY_VALUE
    if stripped.lower().startswith(("signed", "signature", "by:")):
        return BlockType.SIGNATURE
    return default


def normalise_bbox(
    x0: float, y0: float, x1: float, y1: float, *, width: float, height: float
) -> BoundingBox:
    """Convert absolute pixel coordinates to the normalised space used everywhere."""
    if width <= 0 or height <= 0:
        raise ValueError("Page width and height must be positive to normalise a bbox.")
    return BoundingBox(
        x0=min(max(x0 / width, 0.0), 1.0),
        y0=min(max(y0 / height, 0.0), 1.0),
        x1=min(max(x1 / width, 0.0), 1.0),
        y1=min(max(y1 / height, 0.0), 1.0),
    )
