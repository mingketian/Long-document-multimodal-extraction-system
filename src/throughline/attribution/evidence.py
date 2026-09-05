"""Evidence attribution.

A value without a source is an assertion. This module turns the model's claimed
citations into verified pointers - page number, block id, bounding box - and, just
as importantly, marks the ones that do not hold up.

Verification is a three-step ladder, tried in order:

1. **Block id match.** The model named a block that exists in the document. Cheap
   and exact.
2. **Quote match.** The model quoted text; find the block that actually contains it.
   Catches the common failure where the id is hallucinated but the reading was real.
3. **Value search.** Neither id nor quote resolves; look for the extracted value
   itself in the group's pages.

Anything that survives none of the three is recorded as ``UNVERIFIED`` and does not
get to claim a citation. That distinction is what ``citation_precision`` measures:
of the citations the system emits, how many point at text that genuinely supports
the value.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from throughline.ingest.layout import Document, LayoutBlock, Page
from throughline.state.cross_page import EvidenceRef


class AttributionStatus(str, Enum):
    """How an evidence citation was resolved."""

    BLOCK_ID = "block_id"
    QUOTE = "quote"
    VALUE = "value"
    UNVERIFIED = "unverified"

    @property
    def is_verified(self) -> bool:
        return self is not AttributionStatus.UNVERIFIED


@dataclass(frozen=True)
class AttributionResult:
    """One resolved (or unresolved) citation."""

    ref: EvidenceRef | None
    status: AttributionStatus
    detail: str = ""

    @property
    def is_verified(self) -> bool:
        return self.status.is_verified and self.ref is not None


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", _normalise(text))


def _ref_from_block(block: LayoutBlock, quote: str, confidence: float) -> EvidenceRef:
    return EvidenceRef(
        page_number=block.page_number,
        block_id=block.block_id,
        quote=quote or block.text[:200],
        bbox=(block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1),
        confidence=confidence,
    )


def _find_by_quote(quote: str, pages: Sequence[Page]) -> LayoutBlock | None:
    """Locate the block containing a quote, exact first then token-overlap."""
    needle = _normalise(quote)
    if len(needle) < 4:
        return None

    for page in pages:
        for block in page.blocks:
            if needle in _normalise(block.text):
                return block

    # The model paraphrased or the OCR differs slightly; fall back to token overlap.
    needle_tokens = set(_strip_punctuation(quote).split())
    if len(needle_tokens) < 3:
        return None

    best: tuple[float, LayoutBlock] | None = None
    for page in pages:
        for block in page.blocks:
            block_tokens = set(_strip_punctuation(block.text).split())
            if not block_tokens:
                continue
            overlap = len(needle_tokens & block_tokens) / len(needle_tokens)
            if overlap >= 0.7 and (best is None or overlap > best[0]):
                best = (overlap, block)
    return best[1] if best else None


def _find_by_value(value: Any, pages: Sequence[Page]) -> LayoutBlock | None:
    """Locate a block containing the extracted value verbatim."""
    if value in (None, "", [], {}):
        return None
    needle = _normalise(str(value))
    if len(needle) < 3:
        return None

    for page in pages:
        for block in page.blocks:
            if needle in _normalise(block.text):
                return block
    return None


def attribute(
    claim: dict[str, Any],
    value: Any,
    pages: Sequence[Page],
    *,
    document: Document | None = None,
) -> AttributionResult:
    """Resolve one model-supplied citation against the pages it was read from.

    Args:
        claim: The model's evidence entry - ``block_id``, ``quote``, ``confidence``.
        value: The extracted value the citation is meant to support.
        pages: The pages of the group the claim came from.
        document: Optional whole document, allowing a block id from an earlier group
            to still resolve.

    Returns:
        An :class:`AttributionResult`. Callers should treat an unverified result as a
        value with no citation, not as a failed extraction.
    """
    confidence = float(claim.get("confidence", 0.0) or 0.0)
    quote = str(claim.get("quote", "") or "")
    block_id = claim.get("block_id")

    if block_id:
        for page in pages:
            block = page.find_block(str(block_id))
            if block is not None:
                return AttributionResult(
                    _ref_from_block(block, quote, confidence), AttributionStatus.BLOCK_ID
                )
        if document is not None:
            block = document.find_block(str(block_id))
            if block is not None:
                return AttributionResult(
                    _ref_from_block(block, quote, confidence),
                    AttributionStatus.BLOCK_ID,
                    "resolved outside the current page group",
                )

    if quote:
        block = _find_by_quote(quote, pages)
        if block is not None:
            return AttributionResult(
                _ref_from_block(block, quote, confidence * 0.9),
                AttributionStatus.QUOTE,
                f"block id {block_id!r} did not resolve; matched on quoted text",
            )

    block = _find_by_value(value, pages)
    if block is not None:
        return AttributionResult(
            _ref_from_block(block, str(value), confidence * 0.75),
            AttributionStatus.VALUE,
            "matched on the extracted value itself",
        )

    return AttributionResult(
        None,
        AttributionStatus.UNVERIFIED,
        f"no block matched block_id={block_id!r} or quote={quote[:60]!r}",
    )


def attribute_all(
    claims: Iterable[dict[str, Any]],
    value: Any,
    pages: Sequence[Page],
    *,
    document: Document | None = None,
) -> tuple[list[EvidenceRef], list[AttributionResult]]:
    """Resolve every citation for one value.

    Returns the verified refs (possibly empty) and the full result list, so callers
    can both use the good citations and report the bad ones.
    """
    results = [attribute(claim, value, pages, document=document) for claim in claims]
    refs = [result.ref for result in results if result.is_verified and result.ref is not None]
    return refs, results


@dataclass
class AttributionStats:
    """Aggregate attribution quality over a run."""

    total_claims: int = 0
    verified: int = 0
    by_status: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_status is None:
            self.by_status = {status.value: 0 for status in AttributionStatus}

    def record(self, result: AttributionResult) -> None:
        self.total_claims += 1
        self.by_status[result.status.value] += 1
        if result.is_verified:
            self.verified += 1

    @property
    def citation_precision(self) -> float:
        """Fraction of emitted citations that resolve to supporting text."""
        return self.verified / self.total_claims if self.total_claims else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "verified": self.verified,
            "citation_precision": round(self.citation_precision, 4),
            "by_status": dict(self.by_status),
        }
