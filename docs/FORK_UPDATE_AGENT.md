# Fork-Update Agent

**Release automation for the platform the extraction pipeline runs on.**

The second half of the Ricoh USA internship. The first half built a long-document
extraction system on top of the AWS GenAI IDP accelerator; this half keeps the thing
underneath it current, and proves after every change that it still works.

> Source: [`ops/fork_update_agent/`](../ops/fork_update_agent) ·
> Operations: [`FORK_UPDATE_RUNBOOK.md`](FORK_UPDATE_RUNBOOK.md) ·
> Formal design: [`PROCRV_document.pdf`](PROCRV_document.pdf)

![Fork-Update Agent workflow](../results/figures/ops_workflow.svg)

---

## 1. The problem

The extraction pipeline is built on a fork of **`idp_common`**, the shared library
inside the AWS Solutions Library's Intelligent Document Processing accelerator. That
library is not ours and it does not stand still: upstream ships releases regularly, and
those releases can change build behaviour, configuration schemas, or runtime execution.

Keeping the fork current was a manual job, and it had three specific costs.

**It was slow, so it got deferred.** Someone had to notice a release, read the notes,
sync the fork, redeploy the stacks, and then check by hand whether the end-to-end
pipeline still worked. When something broke, diagnosis ran from minutes to days
depending on the size of the upstream delta. The predictable result was that updates
were postponed, and the sandbox drifted further from upstream, which made the eventual
merge larger and riskier — a loop that tightens on itself.

**It failed silently.** The dangerous case is not a crash. It is a configuration
mismatch or a schema change that leaves everything running while quietly degrading
pipeline correctness. Nobody gets paged for that. It surfaces weeks later as a
customer-facing demo producing subtly wrong output.

**It left no trail.** With evidence scattered across pull requests, build logs,
CloudWatch and release notes, "what is currently deployed?" and "when did it change?"
were questions that took an afternoon to answer.

A version number alone is not a sufficient signal either: upstream sometimes introduced
breaking changes without a major version bump.

---

## 2. What it does

An EventBridge rule fires every six hours into a Step Functions state machine over five
Lambdas, each with exactly one operational responsibility.

| # | Lambda | Responsibility |
|---|---|---|
| 01 | `DetectReleaseFn` | Query GitHub `/releases/latest`, falling back to `/tags`. Compare against the version recorded in SSM Parameter Store. |
| 02 | `PrepareMergeFn` | Publish an SNS notification with the release notes and a link to the fork. |
| — | **Human review gate** | A person reviews the changes and clicks "Sync fork". |
| 03 | `DeploySandboxFn` | `UpdateStack` on the sandbox CloudFormation stack, polled to a 30-minute bound. |
| 04 | `RunSmokeTestFn` | Execute the **real** IDP Step Functions workflow on a fixed S3 fixture document. |
| 05 | `ReportStatusFn` | Publish the outcome to SNS with links to the evidence, and record the new version. |

A `Choice` state routes runs where nothing changed straight to a `SKIPPED`
notification — the common case, and it costs one GitHub call and one SSM read.

---

## 3. Four decisions worth defending

### The merge stays behind a human gate

Detection, deployment, validation and reporting are automated. The merge is not.

Automating fork synchronisation requires a GitHub credential with **write access to the
repository**, and a personal access token is the wrong credential for that — it carries
one engineer's full permissions into an unattended system, and it stops working when
that engineer's access changes. The correct credential is a service account, which is
an organisational decision rather than a technical one.

So Phase 1 keeps a person between "upstream released something" and "our sandbox runs
it". The Phase 2 code path exists in `PrepareMergeFn`; enabling it is configuration, not
development. That is a deliberately drawn boundary, not an unfinished feature.

### A smoke test, not a quality benchmark

`RunSmokeTestFn` runs the actual IDP Step Functions workflow on a curated fixture
document and asserts it completes. It does **not** measure extraction accuracy.

That is the right scope for a guardrail. Its job is to catch *breakage* — the pipeline
no longer starts, no longer completes, or throws — quickly and cheaply enough to run on
every update. Accuracy regression is a different question, answered by a labelled
corpus and the evaluation harness ([`EVALUATION.md`](EVALUATION.md)). Fusing the two
would make the guardrail slow and the benchmark unreliable, and neither would be
trusted.

### Three failure classes, reported separately

"The tests failed" is not an actionable message. Every task carries an
`add_catch(result_path="$.error")` that routes to a failure notification tagged with
the stage:

| Class | Means | First thing to check |
|---|---|---|
| `DETECTION` | GitHub unreachable, or no version could be determined | GitHub status; the SSM parameter |
| `DEPLOY` | CloudFormation update failed, rolled back, or timed out | The stack events attached to the notification |
| `SMOKE` | The IDP workflow failed, timed out, or produced invalid output | The execution ARN in the notification — **this is the breakage signal** |

`ReportStatusFn` shapes the message per class, so the notification tells you which
question to ask.

### "No updates to perform" is success

CloudFormation raises a `ValidationError` when a stack update is a no-op.
`DeploySandboxFn` catches that specific case and treats it as success, because a run
where upstream changed something that does not affect our stack is a *correct* run, not
a failed one. Treating it as an error would train the team to ignore failure alerts —
the most expensive possible outcome for a guardrail.

---

## 4. Security posture

| | |
|---|---|
| **Least privilege** | `DetectReleaseFn` reads public GitHub and one SSM parameter. `DeploySandboxFn` can update one named stack. `RunSmokeTestFn` can start and describe Step Functions executions. Nothing has broader reach than its job. |
| **No personal credentials** | Phase 1 needs no GitHub authentication — it only reads a public repository. Phase 2 requires a service account, not a PAT. |
| **Secrets** | SSM Parameter Store, `SecureString`. Never in environment variables, never in code. |
| **Blast radius** | Sandbox account only. The system has no path to production. |
| **Audit** | Every Lambda logs to CloudWatch with 30-day retention; the state machine logs to `/aws/vendedlogs/states/ForkUpdateLogs`. Every SNS notification carries the execution ARN. |

---

## 5. Repository contents

```text
ops/fork_update_agent/
├── infrastructure/cdk/
│   ├── app.py                        CDK app entry point
│   ├── fork_update_agent_stack.py    the whole stack: Lambdas, IAM, SSM, SNS,
│   │                                 the state machine, and the EventBridge rule
│   ├── cdk.json                      context defaults
│   └── requirements.txt
├── source/lambdas/
│   ├── detect_release/handler.py     GitHub releases → SSM comparison
│   ├── prepare_merge/handler.py      SNS notification (Phase 2 PR path present)
│   ├── deploy_sandbox/handler.py     CloudFormation update + bounded polling
│   ├── run_smoke_test/handler.py     IDP workflow execution + polling
│   └── report_status/handler.py      message shaping + version recording
├── state_machines/
│   └── fork_update_agent.asl.json    the ASL definition
├── dev-requirements.txt
└── README.md                         the original project README, as written
```

Plus, at repository root:

- [`docs/FORK_UPDATE_RUNBOOK.md`](FORK_UPDATE_RUNBOOK.md) — deployment, subscription,
  manual triggering, monitoring, and troubleshooting procedures.
- [`docs/PROCRV_document.pdf`](PROCRV_document.pdf) — the formal *Pre-Requirements
  Operating Concept Rationale and Validation* document: scope, current-system analysis,
  justification, the concept for the new system, scenarios, impacts, and the
  alternatives considered and rejected.
- [`tests/test_fork_update_report_status.py`](../tests/test_fork_update_report_status.py) —
  unit tests over `ReportStatusFn`'s message shaping for the SUCCESS, FAILED and SKIPPED
  paths.

The Lambda handlers deliberately use **only `boto3` and the standard library**. A
five-function serverless system does not need a dependency tree, and not having one
removes an entire class of deployment failure.

---

## 6. Alternatives considered and rejected

Recorded in the PROCRV document, and worth restating because the rejections are where
the reasoning is:

| Option | Why not |
|---|---|
| Better documentation, keep the manual process | Does not address inconsistency, silent failures, or the engineering time cost. Documentation does not run. |
| Fully autonomous sync in the MVP | Needs a repository write credential and the governance to go with it. Deferred to Phase 2 as configuration. |
| A full evaluation platform in Phase 1 | Scope beyond a reliable internship-scale MVP. The first system needs to catch breakage; accuracy scoring is a separate, later problem — and is what the extraction repo's evaluation harness now does. |
| Drift detection across configuration schemas | Genuinely valuable, acknowledged as future work. Not a foundation you build first. |

---

## 7. Where it sits in the whole project

```
                     ┌─────────────────────────────────────┐
   upstream          │  AWS GenAI IDP accelerator          │
   idp_common  ─────▶│  (Textract, Step Functions, S3,     │
   releases          │   DynamoDB, CloudWatch, Cognito)    │
        │            └──────────────┬──────────────────────┘
        │                           │
        │  Fork-Update Agent        │  Throughline
        │  keeps this current       │  extends it with a long-document
        │  and verified             │  extraction path
        ▼                           ▼
   detect → notify → [human] → deploy → smoke test → report
                                              │
                                              └──▶ the extraction pipeline
                                                   still works after the update
```

The two halves answer different questions about the same system. Throughline asks
*"can we extract this correctly, and prove it?"* The Fork-Update Agent asks *"is the
platform we extract on still the one we tested against?"* Neither is much use without
the other: an accurate extractor on a silently drifted platform is not accurate for
long, and a perfectly maintained platform with no evaluation harness has nothing to
maintain quality *of*.
