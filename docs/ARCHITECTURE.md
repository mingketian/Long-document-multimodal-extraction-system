# Architecture

How a 40-page document becomes a schema-valid, fully attributed record.

![Architecture](../results/figures/architecture.svg)

---

## The problem in one paragraph

A vision-language model can read a page. It cannot read forty of them at once — not
within a context window, not within a latency budget, and not without the visual
tokens for page 30 crowding out the schema instructions. The obvious workarounds both
fail: feeding one page at a time destroys every relationship that spans a page break
(a table that continues, a total that only reconciles at the end, a defined term used
forty pages after its definition), and truncating the document silently loses whatever
was at the end. **Throughline's answer is a bounded window plus explicit state**: read
a few pages at a time, and carry forward a compact record of what is already known and
what is still missing.

---

## 1. Ingest

**Module:** [`throughline/ingest/`](../src/throughline/ingest)

A page is never just pixels here. Every page carries its rendered image *and* the
OCR/layout blocks extracted from it, because the two are complementary:

| Signal | What it gives you | What it cannot give you |
|---|---|---|
| Page image | Layout, table geometry, stamps, handwriting, checkbox state | Exact characters; a stable address to cite |
| Layout blocks | Exact text, bounding boxes, reading order, a citable `block_id` | Anything the OCR engine failed to read |

That second column is why attribution is possible at all. When the model says
`$41,200`, it also says `p12:b7` — and `p12:b7` is a real object with real coordinates
that can be checked.

Three providers implement one protocol:

| Provider | When |
|---|---|
| `TextractProvider` | Production. `AnalyzeDocument` with `LAYOUT`, `TABLES`, `FORMS` — already wired into the AWS GenAI IDP accelerator. |
| `PyMuPdfProvider` | Digital PDFs. Most enterprise documents carry a text layer; reading it is faster, free, and more accurate than re-OCRing pixels. Raises rather than silently returning an empty page, so the caller can route to real OCR. |
| `JsonFixtureProvider` | Tests and the offline demo. |

`CachedOcrProvider` wraps any of them with a cache keyed on the source bytes, so
re-running a document after a prompt change costs nothing in OCR.

### Reading order

`assign_reading_order()` handles the two-column case explicitly: when block centres
cluster into a left and a right group with a clear gap, the left column is read
through before the right. This matters for attribution — if the model reads in one
order and the system numbers blocks in another, quoted evidence stops lining up.

---

## 2. Partition into page groups

**Module:** [`throughline/grouping/page_groups.py`](../src/throughline/grouping/page_groups.py)

A **page group** is a bounded window of consecutive pages, processed as one unit.

```
document:   [1][2][3][4][5][6][7][8][9][10]
group 0:    [1][2][3][4]
group 1:             [4][5][6][7]        ← page 4 repeats
group 2:                      [7][8][9][10]
```

**Bounded** — at most `max_pages` pages *and* `max_chars` of layout text. Both bounds
matter: four short pages and four dense ones cost very different amounts, and only the
character bound catches the second case.

**Overlapping** — each group repeats the previous group's last page. A row that
straddles the boundary is therefore visible whole in at least one group. The overlap
creates duplicates by construction; §4 explains why that is fine.

**Boundary-aware** — `partition()` scores candidate boundaries and prefers the
least-bad one within the bound. Ending just before a heading costs nothing; ending
mid-table costs a reconciliation. `page_ends_mid_table()` detects the expensive case
from two signals: an explicit continuation phrase, or a table row in the bottom fifth
of the page with no totals line beneath it.

> **A bug worth recording.** The first version of the boundary scorer listed
> `"subtotal"` as a continuation marker. It is not one — a subtotal line ends a short
> table at least as often as it breaks a long one. The effect was that the final totals
> page always looked mid-table, the trimmer pulled the boundary back, and the last page
> of every document was silently dropped. Two fixes: `"subtotal"` came out of the marker
> list, and `partition()` now enforces that a trimmed window must still contain at least
> one page the previous group did not cover. The invariant is asserted, not patched
> around, and `test_partition_always_covers_and_terminates` checks it across eight
> document lengths.

---

## 3. Rank pages against what is missing

**Module:** [`throughline/retrieval/relevant_pages.py`](../src/throughline/retrieval/relevant_pages.py)

Most fields live in a predictable place. An invoice number is on page 1; a
governing-law clause is near the end; a liability cap is wherever "shall not exceed"
appears. Scanning forty pages for a value that occurs on one of them is waste.

Scoring combines self-contained BM25 over each page's layout text with a small
positional prior taken from the field's `page_hint`. The prior is deliberately small —
a hint is a prior, not a filter, and a liability cap that really does sit on page 3
must still be findable when the hint says "near the end".

Groups are scored by the **mean** over the new pages they contribute, not the sum;
summing would reward a group for its size rather than its content.

**Reordering is skipped for schemas that declare tables.** Table rows only accumulate
correctly in document order, so for those schemas page order is a correctness
requirement, not a default.

---

## 4. Cross-page state

**Module:** [`throughline/state/cross_page.py`](../src/throughline/state/cross_page.py)

The object the whole system turns on. It holds what has been extracted, where each
value came from, and what is still outstanding.

### Field resolution

| Situation | Rule | Why |
|---|---|---|
| First observation | Take it | Nothing to compare against |
| Same value again | Corroborate; raise confidence, keep `revision_count` at 0 | A second group agreeing is evidence, not a revision |
| Higher confidence | Replace, increment `revision_count`, log a note | The better-supported reading wins |
| Lower confidence | Keep the existing value | |
| Equal confidence, `continues_across_pages` | **Prefer the later reading** | `total_amount` on page 1 is a partial subtotal; on the last page it is the answer |
| Equal confidence, otherwise | Keep the first | Header fields do not improve by being re-read |

`revision_count` is worth watching: a field revised repeatedly is genuinely ambiguous
in the document, and that is a fact about the document worth surfacing.

### Table accumulation

Rows append and deduplicate on `row_key_columns`, normalised for case and whitespace.
This is what makes the page overlap safe: the boundary row appears in two groups and
collapses to one. A table without declared key columns falls back to whole-row
matching — correct, just less forgiving.

### The carry-over

`render_carry_over()` produces the compact summary injected into the next group's
prompt. It is deliberately lossy, and bounded at 1,800 characters by default. **A
carry-over that grows with the document would defeat the bound page grouping exists to
enforce.** It reports three things:

1. what is settled, with citations;
2. which tables are still open, and that a repeated column header is not a new row;
3. what is still missing — which is what lets the next prompt be *steered*.

---

## 5. Prompt assembly

**Module:** [`throughline/prompting/templates.py`](../src/throughline/prompting/templates.py)

Four signals are fused into one prompt: the schema, the page images, the
block-addressed layout text, and the carry-over. The output contract asks for a JSON
envelope with a parallel evidence map:

```json
{
  "fields":      { "invoice_number": "INV-20260001" },
  "evidence":    { "invoice_number": [{ "block_id": "p1b2", "quote": "Invoice No: INV-20260001", "confidence": 0.97 }] },
  "tables":      { "line_items": [{ "values": {...}, "block_id": "p3b12", "page": 3 }] },
  "open_tables": ["line_items"]
}
```

`open_tables` is the model's own signal that a table is still running at the last page
of the group — the input the early-exit policy reads before deciding it is safe to stop.

---

## 6. Schema-constrained decoding

**Module:** [`throughline/decoding/constrained.py`](../src/throughline/decoding/constrained.py)

"Schema-constrained" means two things depending on the serving stack, and both are
implemented:

**Hard constraint.** `build_grammar()` compiles the schema to JSON Schema for
grammar-constrained decoding — `guided_json` in vLLM, `json_schema` in SGLang, and the
equivalent in Outlines and TGI. The whole envelope is constrained, not just the record,
so the evidence map is as guaranteed as the values it supports.

**Soft constraint.** A hosted endpoint often does not support grammars, so the output
is parsed defensively: markdown fences stripped, the outermost balanced object located
(ignoring braces inside strings), trailing commas removed, bare keys quoted, Python
literals normalised, and — importantly — a **truncated generation salvaged** by closing
what is still open. Long tabular decoding gets cut off; losing the whole group over a
missing brace is far too expensive.

Every repair is recorded in `ParsedEnvelope.repairs`, so how often the soft path is
doing work is a measurable thing rather than a guess.

---

## 7. Evidence attribution

**Module:** [`throughline/attribution/evidence.py`](../src/throughline/attribution/evidence.py)

A three-step ladder, tried in order:

| Step | Method | What it catches |
|---|---|---|
| 1 | Block id | The model named a real block. Exact and cheap. |
| 2 | Quote | The id was wrong but the quoted text is really on the page — a hallucinated address over a genuine reading. |
| 3 | Value | Neither resolves; find the extracted value verbatim. |

Anything surviving none of the three is `UNVERIFIED`. **The value is kept but earns no
citation**, its confidence is cut to 20%, and a note is recorded. Required fields
without verified evidence block early exit — the system would rather read three more
pages than emit a confident value it cannot point at.

`citation_precision` = verified / emitted. It is the metric that separates "right" from
"right for a checkable reason".

---

## 8. Early exit

**Module:** [`throughline/pipeline/early_exit.py`](../src/throughline/pipeline/early_exit.py)

Stop when every required key is **present**, **schema-valid**, and **evidenced** — with
one guard that does most of the work:

> **Never stop while a table is open.** A naive "stop when the schema is satisfied"
> policy looks excellent on header fields and quietly loses half a line-item table.
> `respect_open_tables` is on by default, and the invoice corpus in `examples/` exists
> partly to prove it: because `line_items` is required and stays open until the last
> page, early exit correctly *never fires* there. All three profiles read 100% of the
> pages. That is not the policy failing — it is the policy working.

`patience` requires consecutive unproductive groups before declaring the document
exhausted, because an overlap region legitimately produces nothing new.

Every decision is returned with its reason and retained in `policy.history`, so a run's
stopping point is auditable.

---

## 9. Orchestration

**Module:** [`throughline/pipeline/orchestrator.py`](../src/throughline/pipeline/orchestrator.py)

```
partition → [order groups] → for each group:
    build prompt (schema + images + layout + carry-over + focus fields)
    → generate (through the prompt cache)
    → parse and repair
    → verify every citation
    → merge into cross-page state
    → ask the exit policy
→ validate the accumulated record → ExtractionResult
```

`ExtractionResult` carries the record, the validation report, the full cross-page
state, a per-group `GroupTrace` (tokens, latency, cache hit, fields added, rows added,
citations claimed vs verified, repairs, errors), the exit reason, and attribution
statistics. Everything needed to answer "why did it say that?" and "why did it stop?"

A backend error on one group is recorded on the trace and the run continues; a failing
document in a batch is isolated. `fail_fast=True` inverts both for debugging.

---

## 10. Where the latency goes

Three mechanisms, in the order they pay off:

1. **OCR and prompt caches** — content-addressed on source bytes and on
   prompt+config+backend. Iterating on the merge policy, the exit threshold or the
   attribution ladder changes no prompt, so re-runs are free.
2. **Bounded page groups** — cost per group is predictable, and a group is the unit
   that can be skipped.
3. **Validation-driven early exit** — the run stops when the schema is satisfied, and
   relevant-page retrieval helps it get there sooner by looking in the right place first.

What this repository measures on its own synthetic corpus is in
[`EVALUATION.md`](EVALUATION.md).
