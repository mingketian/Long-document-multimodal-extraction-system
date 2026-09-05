# Deployment and MLOps

The extraction pipeline is half the system. The other half is the machinery that gets a
model onto SageMaker, keeps the platform underneath it current, and proves after every
change that the thing still works.

---

## 1. Where this runs

The pipeline was built inside the **AWS GenAI IDP accelerator** — the AWS Solutions
Library's intelligent-document-processing stack — extending it rather than replacing
it. That decision shapes everything below: Textract, S3, Step Functions, DynamoDB and
CloudWatch were already there, so the work was to add a long-document extraction path,
not to build a document platform.

```
S3 (documents)
   └─> Textract  AnalyzeDocument: LAYOUT + TABLES + FORMS
        └─> Throughline
              partition → per-group prompt → SageMaker (Qwen2.5-VL-7B + LoRA)
              → constrained parse → attribute → cross-page state → early exit
                   └─> S3 (records + evidence maps)   DynamoDB (run metadata)
                        └─> MLflow (params, metrics, per-key F1)
```

---

## 2. Serving the model

**Module:** [`throughline/models/sagemaker.py`](../src/throughline/models/sagemaker.py)

Two endpoint shapes, because the workload has two shapes.

| | `SageMakerBackend` | `SageMakerAsyncBackend` |
|---|---|---|
| Endpoint | Real-time | Asynchronous, via S3 |
| Use | Interactive extraction, one group per request | Corpus processing in batches |
| Payload | Bounded by the real-time limit | Much larger — four page images fit comfortably |
| Cost when idle | Warm instance | **Scales to zero** |
| Retry | Exponential backoff, 3 attempts | Bounded polling to a timeout |

For a corpus that arrives in weekly drops rather than continuously, the async endpoint
is the difference between paying for a GPU all week and paying for the hours it runs.

Both accept an OpenAI-style chat payload so the same handler serves a HuggingFace TGI
container and a custom Qwen2.5-VL container without a translation layer.
`_extract_text()` tolerates the several response shapes SageMaker containers return.

### Deployment

```python
from throughline.training.sagemaker_launch import EndpointDeployment

EndpointDeployment(
    model_data_s3_uri="s3://.../lora-adapter.tar.gz",
    role_arn="arn:aws:iam::...:role/ThroughlineExecution",
    endpoint_name="throughline-qwen25vl",
    instance_type="ml.g5.2xlarge",
    async_inference=True,
    async_output_s3_uri="s3://.../async-output/",
).deploy()
```

---

## 3. Training

**Modules:** [`training/dataset.py`](../src/throughline/training/dataset.py),
[`training/lora.py`](../src/throughline/training/lora.py),
[`training/sagemaker_launch.py`](../src/throughline/training/sagemaker_launch.py)

### The training unit is a page-group set

Not a document, and not a page. Training on whole documents teaches the model nothing
about continuation, because it never sees a boundary. Training on single pages teaches
it that tables end at page breaks — the error we most need it not to make.

Each example is built by **replaying the real pipeline's grouping** over a labelled
document: for every group, given the carry-over state a real run would have had at that
point, what should the model output for this window? Training data and inference data
are the same shape by construction.

Two details carry most of the value:

- **Carry-over is simulated, not idealised.** The state fed to group *n* holds only what
  groups 0..*n*−1 could actually have produced.
- **A field is taught once.** If group 0 settles `invoice_number`, group 1's target
  omits it — which demonstrates the "do not repeat what is settled" rule rather than
  merely stating it in the prompt.

```bash
throughline build-dataset examples/corpus --schema invoice --out data/train.jsonl
```

Splits are **by document, never by example** — splitting by example would put group 0
in train and group 1 in validation, and the carry-over would leak the answer straight
across the boundary.

### LoRA, and three defensible choices

| Choice | Rationale |
|---|---|
| **LoRA, not full fine-tuning** | An order of magnitude less compute, adapters measured in tens of MB, one shared base checkpoint across every schema, and no risk of degrading the general document understanding that made the base worth starting from. |
| **Vision tower frozen** | The task is not "learn to see documents" — the base model already can. It is "emit this schema, from these page groups, with these citations". Adapting the vision encoder spends parameters where the bottleneck is not. |
| **Loss on the completion only** | The prompt carries the schema and the full page layout. Training the model to reproduce text it will always be handed wastes most of the gradient. `CompletionOnlyCollator` masks everything up to the assistant turn. |

Working configuration: r=16 (8 underfits table continuation; 32 showed no gain worth
the memory), α=32, dropout 0.05, attention + MLP projections, sequence length 8,192,
effective batch 16 via gradient accumulation.

```python
from throughline.training.sagemaker_launch import SageMakerTrainingJob

SageMakerTrainingJob(
    role_arn="arn:aws:iam::...:role/ThroughlineTraining",
    train_s3_uri="s3://.../page-group-sets/",
    output_s3_uri="s3://.../adapters/",
    instance_type="ml.g5.2xlarge",
    use_spot=True,      # safe: the trainer checkpoints, so an interruption resumes
).launch()
```

---

## 4. Tracking

**Module:** [`evaluation/mlflow_tracking.py`](../src/throughline/evaluation/mlflow_tracking.py)

Every evaluation run and every training job logs params, metrics and artifacts. Per-key
F1 is logged as its own metric (`f1/<key>`) alongside the headline, so a dashboard can
chart the field that regressed rather than the average that moved.

MLflow is optional. Without it every function degrades to a structured log line.

---

## 5. Keeping the platform underneath current

**Directory:** [`ops/fork_update_agent/`](../ops/fork_update_agent) ·
**Runbook:** [`FORK_UPDATE_RUNBOOK.md`](FORK_UPDATE_RUNBOOK.md) ·
**Design document:** [`PROCRV_document.pdf`](PROCRV_document.pdf)

The extraction pipeline sits on a fork of `idp_common`, the shared library inside the
AWS GenAI IDP accelerator. Upstream moves often and sometimes breaks build behaviour,
configuration schemas, or runtime execution. Keeping the fork current by hand was slow,
easy to defer, and prone to **silent failures** — a configuration mismatch that does not
crash anything but quietly degrades pipeline correctness.

The **Fork-Update Agent** is a Step Functions workflow over five Lambdas that
standardises the loop:

```
EventBridge (every 6h)
  └─> DetectReleaseFn    GitHub releases API, falls back to /tags; compares against SSM
       ├─ no change ──────────────────────────────────> ReportStatusFn (SKIPPED)
       └─ new release
            └─> PrepareMergeFn    SNS notification with release notes
                 │
                 ▼  human review gate — a person clicks "Sync fork"
                 └─> DeploySandboxFn   CloudFormation stack update, 30-min poll
                      └─> RunSmokeTestFn   the real IDP Step Functions workflow,
                      │                     on curated S3 fixture documents
                      └─> ReportStatusFn   SNS + update the recorded version in SSM
```

### Why a human gate

Phase 1 keeps a person between "upstream released something" and "our sandbox runs it".
Fully automated fork synchronisation needs a GitHub service account with write access,
and a personal access token is the wrong credential for that. The code path for
automated PR creation exists in `PrepareMergeFn`; enabling it is configuration, not
development.

### Why a smoke test rather than a quality benchmark

The guardrail's job is to catch **breakage**, not to measure accuracy. It runs the real
Step Functions workflow on a fixed fixture document and asserts it completes. Accuracy
regression testing is the evaluation harness's job, on a labelled corpus, and mixing
the two would make both slower and neither trustworthy.

The design document classifies failures into three kinds so triage is unambiguous:
build-level, deployment-level, and end-to-end validation. `ReportStatusFn` reports which
occurred.

### Least privilege

Each Lambda has only the permissions it needs. Detection reads public GitHub and one
SSM parameter. Deployment can update one named stack. Nothing touches production.
Secrets live in SSM Parameter Store as `SecureString`. CloudWatch retains 30 days.

---

## 6. Operating

| Task | Command |
|---|---|
| Extract one document | `throughline extract doc.pdf --schema invoice --backend sagemaker --endpoint <name>` |
| Evaluate a corpus | `throughline evaluate corpus/ --schema invoice --profile balanced` |
| Compare profiles | `throughline sweep corpus/ --schema invoice` |
| Inspect grouping and page relevance | `throughline inspect doc.json --schema invoice` |
| Cache size / clear | `throughline cache --dir .cache/prompts [--clear]` |

Environment overrides: `THROUGHLINE_BACKEND`, `THROUGHLINE_ENDPOINT`,
`THROUGHLINE_ADAPTER`, `THROUGHLINE_SCHEMA`, `THROUGHLINE_REGION`, `THROUGHLINE_CACHE`.

### Choosing a profile

| Profile | Stops when | Use for |
|---|---|---|
| `accuracy` | Never — reads every group in page order | Measuring the ceiling; any document of record |
| `balanced` | Schema satisfied, valid **and** evidenced | Production default |
| `fast` | Schema satisfied and valid; evidence not required | Triage and search indexing — **not** a system of record |
