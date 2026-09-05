"""Relevant-page retrieval.

Reading every page group for every field is the obvious thing to do and the wrong
one. Most fields in a long document live in a predictable place: an invoice number
is on page 1, a governing-law clause is near the end, a liability cap is wherever
the word "exceed" appears. Scanning all 40 pages to find a value that only ever
occurs on one of them is pure waste.

This module scores pages against the *fields that are still missing* and lets the
orchestrator visit promising groups first. Combined with validation-driven early
exit, that is where most of the latency reduction comes from: the run stops once
the schema is satisfied, and it gets there sooner because it looked in the right
place first.

Scoring is deliberately lexical (BM25 over layout text plus positional priors).
A dense retriever is available behind the same interface, but for schema fields -
whose anchors are short, literal strings like "Invoice No." - lexical matching is
both stronger and free.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from throughline.grouping.page_groups import PageGroup
from throughline.ingest.layout import Document, Page
from throughline.schema.spec import ExtractionSchema, FieldSpec, TableSpec

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those", "which"]
)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


@dataclass
class PageScore:
    """One page's relevance to the currently-missing fields."""

    page_number: int
    score: float
    matched_fields: tuple[str, ...] = ()

    def __lt__(self, other: PageScore) -> bool:  # enables sorted()
        return self.score < other.score


class BM25Index:
    """Small self-contained BM25 over a document's pages.

    Self-contained on purpose: a page index for a 40-page document is a handful of
    dictionaries, and depending on a search library for that would be heavier than
    the thing it replaces.
    """

    __slots__ = ("_docs", "_page_numbers", "_df", "_avg_len", "_k1", "_b", "_n")

    def __init__(self, pages: Sequence[Page], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs: list[Counter[str]] = []
        self._page_numbers: list[int] = []

        for page in pages:
            self._docs.append(Counter(tokenize(page.text)))
            self._page_numbers.append(page.page_number)

        self._n = len(self._docs)
        self._df: Counter[str] = Counter()
        for doc in self._docs:
            self._df.update(doc.keys())
        total = sum(sum(doc.values()) for doc in self._docs)
        self._avg_len = total / self._n if self._n else 0.0

    def score(self, query_tokens: Sequence[str]) -> dict[int, float]:
        """BM25 score per page number for one bag of query tokens."""
        if not self._n or not query_tokens:
            return {}

        scores: dict[int, float] = {}
        for index, doc in enumerate(self._docs):
            length = sum(doc.values()) or 1
            total = 0.0
            for token in query_tokens:
                frequency = doc.get(token, 0)
                if not frequency:
                    continue
                df = self._df.get(token, 0)
                idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / (self._avg_len or 1)
                )
                total += idf * frequency * (self._k1 + 1) / denominator
            if total:
                scores[self._page_numbers[index]] = total
        return scores


def _field_query(spec: FieldSpec | TableSpec) -> list[str]:
    """The lexical anchors for one schema entry."""
    parts: list[str] = [spec.name.replace("_", " ")]
    parts.append(spec.description or "")
    if isinstance(spec, FieldSpec):
        parts.extend(spec.keywords)
        parts.append(spec.page_hint)
    else:
        parts.extend(column.name.replace("_", " ") for column in spec.columns)
    return tokenize(" ".join(part for part in parts if part))


_POSITION_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfirst page|cover page|letterhead|header block\b"), "front"),
    (re.compile(r"\blast page|totals block|signature|signatures\b"), "back"),
    (re.compile(r"\bevery page\b"), "any"),
)


def _positional_prior(spec: FieldSpec | TableSpec, page_number: int, page_count: int) -> float:
    """Bonus for pages where a field's ``page_hint`` says it should live.

    Small by design - a hint is a prior, not a filter. A liability cap that really
    does sit on page 3 must still be findable when the hint says "near the end".
    """
    if page_count <= 1 or not isinstance(spec, FieldSpec) or not spec.page_hint:
        return 0.0

    hint = spec.page_hint.lower()
    position = (page_number - 1) / max(page_count - 1, 1)

    for pattern, region in _POSITION_HINTS:
        if not pattern.search(hint):
            continue
        if region == "front":
            return 1.5 * max(0.0, 1.0 - position * 2.5)
        if region == "back":
            return 1.5 * max(0.0, (position - 0.6) / 0.4)
        return 0.25
    return 0.0


@dataclass
class RelevantPageRetriever:
    """Rank a document's pages against a set of target fields."""

    document: Document
    schema: ExtractionSchema
    index: BM25Index = field(init=False)

    def __post_init__(self) -> None:
        self.index = BM25Index(self.document.pages)

    def rank_pages(self, field_names: Iterable[str] | None = None) -> list[PageScore]:
        """Score every page against the named fields (default: the whole schema)."""
        names = list(field_names) if field_names is not None else list(self.schema.all_keys)
        specs: list[FieldSpec | TableSpec] = []
        for name in names:
            spec = self.schema.field(name) or self.schema.table(name)
            if spec is not None:
                specs.append(spec)
        if not specs:
            return []

        page_count = self.document.page_count
        totals: dict[int, float] = {}
        matches: dict[int, set[str]] = {}

        for spec in specs:
            scores = self.index.score(_field_query(spec))
            best = max(scores.values(), default=0.0)
            for page in self.document.pages:
                lexical = scores.get(page.page_number, 0.0)
                prior = _positional_prior(spec, page.page_number, page_count)
                contribution = lexical + prior
                if contribution <= 0:
                    continue
                totals[page.page_number] = totals.get(page.page_number, 0.0) + contribution
                # Only credit a field to a page that is a genuine lexical hit for it.
                if best > 0 and lexical >= 0.55 * best:
                    matches.setdefault(page.page_number, set()).add(spec.name)

        ranked = [
            PageScore(
                page_number=page_number,
                score=score,
                matched_fields=tuple(sorted(matches.get(page_number, ()))),
            )
            for page_number, score in totals.items()
        ]
        ranked.sort(key=lambda item: (-item.score, item.page_number))
        return ranked

    def top_pages(self, field_names: Iterable[str] | None = None, *, k: int = 5) -> list[int]:
        return [item.page_number for item in self.rank_pages(field_names)[:k]]

    def rank_groups(
        self, groups: Sequence[PageGroup], field_names: Iterable[str] | None = None
    ) -> list[tuple[PageGroup, float]]:
        """Order page groups by how promising they are for the named fields.

        A group's score is the mean over the *new* pages it contributes, not the sum:
        summing would reward large groups for their size rather than their content.
        """
        page_scores = {item.page_number: item.score for item in self.rank_pages(field_names)}
        ordered: list[tuple[PageGroup, float]] = []

        for group in groups:
            considered = group.new_page_numbers() or group.page_numbers
            values = [page_scores.get(number, 0.0) for number in considered]
            ordered.append((group, sum(values) / len(values) if values else 0.0))

        ordered.sort(key=lambda item: (-item[1], item[0].group_index))
        return ordered

    def explain(self, field_name: str, *, k: int = 3) -> str:
        """Human-readable trace of why the retriever likes certain pages for a field."""
        ranked = self.rank_pages([field_name])[:k]
        if not ranked:
            return f"{field_name}: no page scored above zero."
        parts = ", ".join(f"p{item.page_number} ({item.score:.2f})" for item in ranked)
        return f"{field_name}: {parts}"
