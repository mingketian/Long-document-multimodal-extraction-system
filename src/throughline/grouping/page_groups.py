"""Bounded page-group partitioning.

A 40-page document does not fit in one VLM context, and feeding it page by page
throws away the continuity that makes long documents hard in the first place. The
compromise is a *page group*: a bounded window of consecutive pages, processed as a
unit, with a small overlap into the previous window.

Two properties matter.

**Bounded.** Every group is at most ``max_pages`` pages and at most ``max_chars``
characters of layout text. That bound is what makes cost per group predictable, and
it is what lets the orchestrator stop early without leaving a half-read window.

**Overlapping.** Each group after the first repeats the last ``overlap`` page(s) of
its predecessor. A table row that straddles the boundary is therefore visible whole
in at least one group, and the merge step can recognise the repeat by row key rather
than guessing.

Group boundaries also prefer to fall where the document itself breaks. If a page
ends a table and the next starts a new section, splitting there costs nothing;
splitting mid-table costs a reconciliation. :func:`partition` scores candidate
boundaries and picks the least-bad one inside the bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from throughline.ingest.layout import BlockType, Document, Page
from throughline.schema.spec import ExtractionSchema, TableSpec


@dataclass(frozen=True)
class PageGroup:
    """A bounded window of consecutive pages."""

    group_index: int
    pages: tuple[Page, ...]
    overlap_pages: tuple[int, ...] = ()
    """Page numbers in this group that were also present in the previous group."""

    continues_table: bool = False
    """True when the previous group ended mid-table."""

    def __len__(self) -> int:
        return len(self.pages)

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(page.page_number for page in self.pages)

    @property
    def first_page(self) -> int:
        return self.pages[0].page_number

    @property
    def last_page(self) -> int:
        return self.pages[-1].page_number

    @property
    def is_overlap_only(self) -> bool:
        """A group that adds no new pages carries no new information."""
        return len(self.overlap_pages) == len(self.pages)

    def new_page_numbers(self) -> tuple[int, ...]:
        overlap = set(self.overlap_pages)
        return tuple(n for n in self.page_numbers if n not in overlap)

    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)

    def render_layout(self, *, max_chars_per_page: int = 6_000) -> str:
        """Layout text for every page in the group, page-delimited for the prompt."""
        sections = []
        for page in self.pages:
            marker = " (repeated from previous group)" if page.page_number in self.overlap_pages else ""
            sections.append(
                f"--- PAGE {page.page_number}{marker} ---\n"
                f"{page.render_layout(max_chars=max_chars_per_page)}"
            )
        return "\n\n".join(sections)

    def image_paths(self) -> list[str]:
        return [page.image_path for page in self.pages if page.image_path]

    def describe(self) -> str:
        span = (
            f"page {self.first_page}"
            if self.first_page == self.last_page
            else f"pages {self.first_page}-{self.last_page}"
        )
        notes = []
        if self.overlap_pages:
            notes.append(f"overlap {list(self.overlap_pages)}")
        if self.continues_table:
            notes.append("table continues")
        return f"group {self.group_index} ({span}){' · ' + ', '.join(notes) if notes else ''}"


@dataclass(frozen=True)
class GroupingConfig:
    """Bounds and preferences for partitioning."""

    max_pages: int = 4
    """Hard upper bound on pages per group."""

    overlap: int = 1
    """Pages repeated from the previous group. 0 disables overlap."""

    max_chars: int = 18_000
    """Hard upper bound on layout characters per group."""

    min_pages: int = 1
    """Never emit a group smaller than this unless the document itself is smaller."""

    prefer_natural_breaks: bool = True
    """Shrink a group below max_pages when a cleaner boundary is available."""

    def __post_init__(self) -> None:
        if self.max_pages < 1:
            raise ValueError("max_pages must be >= 1.")
        if not 0 <= self.overlap < self.max_pages:
            raise ValueError("overlap must be >= 0 and < max_pages.")
        if self.min_pages < 1 or self.min_pages > self.max_pages:
            raise ValueError("min_pages must be between 1 and max_pages.")


def _table_markers(schema: ExtractionSchema | None) -> tuple[str, ...]:
    if schema is None:
        return TableSpec.__dataclass_fields__["continuation_markers"].default
    markers: list[str] = []
    for table in schema.tables:
        markers.extend(table.continuation_markers)
    return tuple(dict.fromkeys(markers)) or TableSpec.__dataclass_fields__[
        "continuation_markers"
    ].default


def page_ends_mid_table(page: Page, schema: ExtractionSchema | None = None) -> bool:
    """True when a page looks like it stops in the middle of a table.

    Two signals, either of which is enough: an explicit continuation phrase
    ("continued on next page", "carried forward"), or a table row sitting in the
    bottom fifth of the page with no totals line beneath it.
    """
    lowered = page.text.lower()
    if any(marker in lowered for marker in _table_markers(schema)):
        return True

    rows = page.blocks_of_type(BlockType.TABLE_ROW, BlockType.TABLE)
    if not rows:
        return False

    lowest = max(row.bbox.y1 for row in rows)
    if lowest < 0.80:
        return False

    below = [
        block
        for block in page.blocks
        if block.bbox.y0 >= lowest and block.block_type is not BlockType.FOOTER
    ]
    totals_words = ("total", "subtotal", "balance", "sum")
    has_totals = any(word in block.text.lower() for block in below for word in totals_words)
    return not has_totals


def _boundary_penalty(page: Page, schema: ExtractionSchema | None) -> float:
    """How costly it is to end a group after this page. Lower is better."""
    if page_ends_mid_table(page, schema):
        return 1.0

    blocks = page.blocks
    if not blocks:
        return 0.0

    # Ending just before a heading is the cleanest possible split.
    tail = sorted(blocks, key=lambda b: b.bbox.y1)[-3:]
    if any(block.block_type in {BlockType.HEADING, BlockType.TITLE} for block in tail):
        return 0.1

    # A page whose last block is a paragraph may be mid-sentence.
    last = max(blocks, key=lambda b: b.bbox.y1)
    if last.block_type is BlockType.PARAGRAPH and not last.text.rstrip().endswith((".", "!", "?", ":")):
        return 0.5
    return 0.2


def partition(
    document: Document,
    config: GroupingConfig | None = None,
    schema: ExtractionSchema | None = None,
) -> list[PageGroup]:
    """Split a document into bounded, overlapping page groups.

    Args:
        document: The document to split.
        config: Bounds and preferences. Defaults to 4 pages with 1 page of overlap.
        schema: Optional schema, used only to source table continuation markers.

    Returns:
        Groups in page order. Every page appears in at least one group.
    """
    config = config or GroupingConfig()
    pages = sorted(document.pages, key=lambda page: page.page_number)
    if not pages:
        return []

    groups: list[PageGroup] = []
    cursor = 0
    group_index = 0

    while cursor < len(pages):
        overlap_start = max(0, cursor - config.overlap) if groups else cursor
        window: list[Page] = []
        used_chars = 0

        for position in range(overlap_start, len(pages)):
            page = pages[position]
            page_chars = len(page.text)

            over_pages = len(window) >= config.max_pages
            over_chars = used_chars + page_chars > config.max_chars and len(window) >= config.min_pages
            if over_pages or over_chars:
                break

            window.append(page)
            used_chars += page_chars

        if not window:  # a single page larger than max_chars still has to go somewhere
            window = [pages[overlap_start]]

        # A trimmed window must still contribute at least one page the previous
        # group did not already cover, or the partition stalls and silently drops
        # the tail of the document.
        first_new = next(
            (index for index, page in enumerate(window) if page.page_number >= pages[cursor].page_number),
            0,
        )
        min_length = max(config.min_pages, first_new + 1)

        if config.prefer_natural_breaks and len(window) > min_length:
            window = _trim_to_natural_break(window, config, schema, min_length=min_length)

        overlap_numbers = tuple(
            page.page_number
            for page in window
            if groups and page.page_number <= groups[-1].last_page
        )
        continues = bool(groups) and page_ends_mid_table(
            pages[cursor - 1] if cursor > 0 else window[0], schema
        )

        groups.append(
            PageGroup(
                group_index=group_index,
                pages=tuple(window),
                overlap_pages=overlap_numbers,
                continues_table=continues,
            )
        )
        group_index += 1

        advanced = window[-1].page_number
        next_cursor = next(
            (i for i, page in enumerate(pages) if page.page_number > advanced), len(pages)
        )
        # min_length above guarantees progress; this assertion documents the
        # invariant rather than papering over a violation of it by skipping a page.
        assert next_cursor > cursor, (
            f"page grouping failed to advance at page {pages[cursor].page_number}"
        )
        cursor = next_cursor

    return groups


def _trim_to_natural_break(
    window: Sequence[Page],
    config: GroupingConfig,
    schema: ExtractionSchema | None,
    *,
    min_length: int = 1,
) -> list[Page]:
    """Shorten a full-size window when an earlier page is a cleaner place to stop.

    ``min_length`` is the floor the caller needs for the window to still advance past
    the previous group; trimming below it would stall the partition.
    """
    best_length = len(window)
    best_penalty = _boundary_penalty(window[-1], schema)

    for length in range(max(config.min_pages, min_length), len(window)):
        penalty = _boundary_penalty(window[length - 1], schema)
        # Require a clear improvement; otherwise keep the larger, cheaper-per-page group.
        if penalty + 0.25 < best_penalty:
            best_penalty = penalty
            best_length = length

    return list(window[:best_length])


def summarise(groups: Sequence[PageGroup]) -> str:
    """One line per group, for logs and the CLI."""
    return "\n".join(group.describe() for group in groups)
