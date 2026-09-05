"""Page-group partitioning."""

from __future__ import annotations

import pytest

from throughline.grouping.page_groups import (
    GroupingConfig,
    page_ends_mid_table,
    partition,
)
from throughline.ingest.layout import BlockType, BoundingBox, Document, LayoutBlock, Page
from throughline.schema import registry


def make_page(number: int, *specs: tuple[str, str, float, float]) -> Page:
    page = Page(page_number=number, width=1240, height=1754)
    for index, (block_type, text, y0, y1) in enumerate(specs):
        page.blocks.append(
            LayoutBlock(
                block_id=f"p{number}b{index}",
                page_number=number,
                block_type=BlockType(block_type),
                text=text,
                bbox=BoundingBox(0.08, y0, 0.92, y1),
                reading_order=index,
            )
        )
    return page


def make_document(page_count: int, **kwargs) -> Document:
    return Document(
        document_id="doc",
        pages=[
            make_page(n, ("paragraph", f"Body text on page {n}. It ends properly.", 0.1, 0.2))
            for n in range(1, page_count + 1)
        ],
        **kwargs,
    )


class TestPartition:
    def test_every_page_appears_in_some_group(self) -> None:
        document = make_document(11)
        groups = partition(document, GroupingConfig(max_pages=4, overlap=1))

        covered = {page for group in groups for page in group.page_numbers}
        assert covered == set(range(1, 12))

    @pytest.mark.parametrize("page_count", [1, 2, 3, 5, 8, 13, 21, 40])
    def test_partition_always_covers_and_terminates(self, page_count: int) -> None:
        """The bug this guards: a trimmed window that fails to advance drops pages."""
        document = make_document(page_count)
        groups = partition(document, GroupingConfig(max_pages=4, overlap=1))

        covered = {page for group in groups for page in group.page_numbers}
        assert covered == set(range(1, page_count + 1))
        assert len(groups) <= page_count

    def test_groups_respect_max_pages(self) -> None:
        groups = partition(make_document(20), GroupingConfig(max_pages=3, overlap=1))
        assert all(len(group) <= 3 for group in groups)

    def test_overlap_repeats_the_previous_last_page(self) -> None:
        groups = partition(make_document(9), GroupingConfig(max_pages=3, overlap=1))
        for previous, current in zip(groups, groups[1:], strict=False):
            assert previous.last_page in current.page_numbers
            assert current.overlap_pages

    def test_zero_overlap_produces_disjoint_groups(self) -> None:
        groups = partition(make_document(9), GroupingConfig(max_pages=3, overlap=0))
        seen: list[int] = []
        for group in groups:
            seen.extend(group.page_numbers)
        assert len(seen) == len(set(seen)) == 9

    def test_new_page_numbers_excludes_overlap(self) -> None:
        groups = partition(make_document(9), GroupingConfig(max_pages=3, overlap=1))
        assert groups[0].new_page_numbers() == groups[0].page_numbers
        for group in groups[1:]:
            assert set(group.new_page_numbers()).isdisjoint(group.overlap_pages)

    def test_empty_document_yields_no_groups(self) -> None:
        assert partition(Document(document_id="empty")) == []

    def test_single_page_document(self) -> None:
        groups = partition(make_document(1))
        assert len(groups) == 1
        assert groups[0].page_numbers == (1,)

    def test_char_budget_shrinks_groups(self) -> None:
        document = Document(
            document_id="fat",
            pages=[
                make_page(n, ("paragraph", "x" * 9_000, 0.1, 0.9)) for n in range(1, 7)
            ],
        )
        groups = partition(document, GroupingConfig(max_pages=4, overlap=0, max_chars=10_000))
        assert all(group.char_count() <= 19_000 for group in groups)
        assert {p for g in groups for p in g.page_numbers} == set(range(1, 7))


class TestGroupingConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_pages": 0},
            {"max_pages": 3, "overlap": 3},
            {"max_pages": 3, "overlap": -1},
            {"max_pages": 3, "min_pages": 4},
        ],
    )
    def test_invalid_configs_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            GroupingConfig(**kwargs)


class TestMidTableDetection:
    def test_explicit_continuation_marker(self) -> None:
        page = make_page(
            1,
            ("table_row", "1  Widget  2  $4.00  $8.00", 0.4, 0.44),
            ("page_footer", "Continued on next page", 0.95, 0.98),
        )
        assert page_ends_mid_table(page, registry.get("invoice"))

    def test_totals_line_means_the_table_ended(self) -> None:
        page = make_page(
            1,
            ("table_row", "9  Widget  2  $4.00  $8.00", 0.80, 0.84),
            ("key_value", "Total Amount Due: $412.00", 0.86, 0.90),
        )
        assert not page_ends_mid_table(page, registry.get("invoice"))

    def test_subtotal_alone_is_not_a_continuation_signal(self) -> None:
        """A subtotal line ends a short table as often as it breaks a long one."""
        page = make_page(1, ("key_value", "Subtotal: $1,204.00", 0.30, 0.34))
        assert not page_ends_mid_table(page, registry.get("invoice"))

    def test_row_at_the_page_foot_with_nothing_after_it(self) -> None:
        page = make_page(1, ("table_row", "12  Widget  2  $4.00  $8.00", 0.86, 0.90))
        assert page_ends_mid_table(page, registry.get("invoice"))

    def test_page_without_a_table_is_never_mid_table(self) -> None:
        page = make_page(1, ("paragraph", "Nothing tabular here at all.", 0.1, 0.2))
        assert not page_ends_mid_table(page, registry.get("invoice"))
