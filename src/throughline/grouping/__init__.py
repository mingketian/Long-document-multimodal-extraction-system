"""Bounded, overlapping page groups."""

from throughline.grouping.page_groups import (
    GroupingConfig,
    PageGroup,
    page_ends_mid_table,
    partition,
    summarise,
)

__all__ = ["GroupingConfig", "PageGroup", "page_ends_mid_table", "partition", "summarise"]
