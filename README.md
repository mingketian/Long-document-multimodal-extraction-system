<div align="center">

# Throughline

**Cross-page state for long-document multimodal extraction.**

A schema-constrained extraction pipeline for documents too long to fit in one context
window. Pages are partitioned into bounded, overlapping groups; a cross-page state
carries extracted fields and evidence references from one group to the next; every
value is attributed back to the block it was read from; and the run stops as soon as
the schema is satisfied and evidenced.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Model](https://img.shields.io/badge/model-Qwen2.5--VL--7B-6f42c1)](src/throughline/models/qwen_vl.py)
[![AWS](https://img.shields.io/badge/serving-SageMaker-FF9900?logo=amazonaws&logoColor=white)](src/throughline/models/sagemaker.py)
[![Tracking](https://img.shields.io/badge/tracking-MLflow-0194E2)](src/throughline/evaluation/mlflow_tracking.py)
[![Tests](https://img.shields.io/badge/tests-148%20passing-1baf7a)](tests)

[Why](#why-this-exists) · [Quickstart](#quickstart) · [Architecture](#architecture) · [Results](#results) · [Docs](#documentation)

</div>

---

## Why this exists

A vision-language model can read a page. It cannot read forty at once — not within a
context window, not within a latency budget. The two obvious workarounds both fail:

| Workaround | What breaks |
|---|---|
| One page at a time | Every relationship that spans a page break. A table that continues. A total that only reconciles on the last page. A defined term used forty pages after its definition. |
| Truncate the document | Whatever was at the end — which in business documents is the totals, the signatures, and the governing law. |

**Throughline's answer is a bounded window plus explicit state.** Read a few pages at a
time, and carry forward a compact record of what is already known, what is still
missing, and which tables are mid-flight.

Three properties follow from that, and they are what the code is actually about:

- **Continuation is a first-class concept.** A row that straddles a page break is one
  row. A repeated column header is not a new row. The group overlap makes the boundary
  visible; the row key collapses the duplicate.
- **Nothing is trusted without a citation.** Every value names the block it was read
  from, and every citation is verified against the document. Unverified values are kept
  but marked, and they block early exit.
- **Stopping is a policy, not an accident.** The run ends when every required key is
  present, schema-valid and evidenced — and never while a table is still open.

---

## Quickstart

No GPU, no cloud account, no model weights. The repository ships a synthetic corpus and
a deterministic rule-based backend so the whole pipeline runs after a plain install.

```bash
git clone https://github.com/mingketian/Long-document-multimodal-extraction-system.git Throughline
cd Throughline
pip install -e .
```

```bash
# see how a document partitions, and which pages the retriever favours
throughline inspect examples/documents/invoice_0001.json --schema invoice

# extract, with citations
throughline extract examples/documents/invoice_0001.json --schema invoice --show-evidence

# compare the accuracy / coverage profiles on a corpus
throughline sweep examples/corpus_agreements --schema service_agreement
```

```
invoice_0001: schema-valid · 2/2 groups · 5 pages · 100% citations verified · 0.01s

Evidence:
  invoice_number: p1:p1b2
  total_amount:   p5:p5b3
  line_items:     p1:p1b8, p1:p1b9, p1:p1b10
```

In Python:

```python
from throughline import ExtractionPipeline
from throughline.ingest import JsonFixtureProvider
from throughline.models import RuleBasedBackend
from throughline.schema import registry

document = JsonFixtureProvider().extract("examples/documents/invoice_0001.json")
result = ExtractionPipeline(RuleBasedBackend(), registry.get("invoice")).run(document)

print(result.summary())
print(result.record["total_amount"])
print(result.evidence_for("total_amount")[0].cite())   # -> p5:p5b3
```

Against a real model:

```bash
throughline extract contract.pdf --schema service_agreement \
  --backend sagemaker --endpoint throughline-qwen25vl
```

---

## Architecture

![Architecture](results/figures/architecture.svg)

| Stage | Module | What it does |
|---|---|---|
| **Ingest** | [`ingest/`](src/throughline/ingest) | Pages carry both the rendered image *and* OCR layout blocks. The blocks are what make citations addressable. Textract, PyMuPDF, or JSON fixtures behind one protocol. |
| **Partition** | [`grouping/`](src/throughline/grouping) | Bounded (≤4 pages, ≤18k chars), overlapping (1 page) windows. Boundaries prefer natural breaks and avoid cutting mid-table. |
| **Retrieve** | [`retrieval/`](src/throughline/retrieval) | BM25 + positional priors rank pages against the keys still missing, so the promising group is read first. Skipped for table schemas, where page order is a correctness requirement. |
| **Prompt** | [`prompting/`](src/throughline/prompting) | Fuses schema, page images, block-addressed layout text, and carry-over state. |
| **Generate** | [`models/`](src/throughline/models) | Qwen2.5-VL-7B locally or on SageMaker (real-time or async); a rule-based backend for offline runs and CI. |
| **Constrain** | [`decoding/`](src/throughline/decoding) | JSON-Schema grammar where the stack supports it; defensive parse and repair where it does not — including salvaging a truncated generation. |
| **Attribute** | [`attribution/`](src/throughline/attribution) | Block id → quote → value, tried in order. What resolves to nothing is marked unverified, not trusted. |
| **Merge** | [`state/`](src/throughline/state) | Fields resolve by confidence; table rows accumulate and deduplicate on their row key. |
| **Exit** | [`pipeline/`](src/throughline/pipeline) | Stop when the schema is satisfied, valid and evidenced — never while a table is open. |

Full walkthrough: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Results

Two kinds of number, deliberately kept apart.

### Reported outcomes — Ricoh USA deployment

> Measured internally between September and December 2025 on ~600 proprietary
> enterprise documents (3.2K pages) with a LoRA-fine-tuned Qwen2.5-VL-7B on SageMaker.
> **Not reproducible from this repository**, which ships a synthetic corpus and a
> clean-room implementation of the same architecture.

| | Before | After | Δ |
|---|---:|---:|---:|
| Weighted F1 *(LoRA on 2.4K page-group sets)* | 0.867 | **0.923** | +5.6 pts |
| Schema-valid output rate | 0.934 | **0.986** | +5.2 pts |
| Cross-page field accuracy *(orchestration harness)* | 0.789 | **0.887** | +9.8 pts |
| Citation precision | 0.879 | **0.942** | +6.3 pts |
| Processing time *(caching, bounded groups, early exit)* | — | **−20.2%** | at −0.3 F1 |

<p align="center">
<img src="results/figures/reported_fine_tuning.png" width="49%" alt="LoRA fine-tuning results">
<img src="results/figures/reported_orchestration.png" width="49%" alt="Orchestration harness results">
</p>

### Measured here

Produced by `make results` over the synthetic corpus, using the **rule-based baseline
backend** — keyword matching, no model. Absolute values are low by construction; the
shape of the trade-off is the finding.

![Trade-off](results/figures/measured_tradeoff.png)

| Profile | Weighted F1 | Pages read | Schema-valid | Citations verified |
|---|---:|---:|---:|---:|
| `accuracy` | 0.364 | 100.0% | 1.000 | 1.000 |
| `balanced` | 0.281 | 31.6% | 1.000 | 1.000 |
| `fast` | 0.091 | 13.6% | 1.000 | 1.000 |

`balanced` reads **32% of the pages for 77% of the accuracy ceiling**. How good that
trade is depends entirely on backend strength — with a weak extractor the skipped pages
were carrying real information; with the fine-tuned model above, the same mechanism
cost 0.3 F1 for 20.2% less time.

Two results worth reading carefully:

- **`line_items` scores 1.000 on all 12 invoices.** The table spans several page groups
  and the overlap re-shows boundary rows, so a correct row count means cross-page
  accumulation *and* deduplication both worked.
- **On invoices, early exit never fires** — all three profiles read 100% of pages.
  `line_items` is required and stays open until the last page, so stopping would
  truncate it and the policy refuses. A speedup here would mean the guard was broken.

Detail, metric definitions and limitations: [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Repository layout

```text
Throughline/
├── src/throughline/
│   ├── schema/          extraction contracts: fields, tables, validation, repair
│   ├── ingest/          pages, layout blocks, OCR providers (Textract / PyMuPDF)
│   ├── grouping/        bounded, overlapping page groups
│   ├── retrieval/       BM25 + positional relevant-page ranking
│   ├── state/           cross-page state — the object the system turns on
│   ├── prompting/       prompt assembly and the output contract
│   ├── models/          Qwen2.5-VL, SageMaker real-time/async, rule-based baseline
│   ├── decoding/        JSON-Schema grammars and defensive parsing
│   ├── attribution/     citation verification
│   ├── pipeline/        orchestrator + early-exit policy
│   ├── caching/         content-addressed OCR and prompt caches
│   ├── training/        page-group datasets, LoRA, SageMaker launchers
│   └── evaluation/      metrics, harness, MLflow tracking
├── ops/fork_update_agent/   CDK + Step Functions + 5 Lambdas, keeping the fork current
├── examples/            20 synthetic documents (246 pages) + labelled corpora
├── tests/               148 tests
├── tools/               corpus, table and figure generation
├── docs/                architecture, evaluation, deployment, runbook, design doc
└── site/                project website
```

---

## Documentation

| Document | Read it for |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Stage-by-stage walkthrough, with the design decisions and one recorded bug |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Metric definitions, reported vs measured numbers, limitations |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | SageMaker serving, LoRA training, MLflow, and the fork-update agent |
| [`docs/FORK_UPDATE_RUNBOOK.md`](docs/FORK_UPDATE_RUNBOOK.md) | Operational procedures for the release-automation workflow |
| [`docs/PROCRV_document.pdf`](docs/PROCRV_document.pdf) | The formal design document for the fork-update agent |

```bash
make help       # every task
make test       # 148 tests
make results    # regenerate tables and figures
make demo       # inspect + extract + sweep, end to end
```

---

## Context and scope

Built during an AI/ML Engineering internship at **Ricoh USA** (Sep–Dec 2025), extending
the AWS GenAI IDP accelerator with a long-document extraction path.

The code here is a **clean-room implementation** of that architecture, written to be
runnable and testable in the open. It contains no Ricoh source, no customer documents,
and no production configuration. The `ops/fork_update_agent/` directory is the
release-automation component built during the same internship.

Every document in `examples/` is synthetic and generated by
[`tools/make_examples.py`](tools/make_examples.py).

## Licence

Proprietary — all rights reserved. Not for redistribution.
