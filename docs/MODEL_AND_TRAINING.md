# The model: Qwen2.5-VL-7B and LoRA

Why this base model, what its visual tokeniser does to a long document, where the
adapter goes, and what the numbers underneath the hyperparameters actually are.

---

## 1. Why Qwen2.5-VL-7B

The requirement is narrower than "a good VLM". It is: read a page group, produce
schema-shaped JSON, and cite the block each value came from. Three properties of
Qwen2.5-VL matter for that.

**Native dynamic resolution.** Qwen2.5-VL does not squash every image to a fixed
grid. It maps a page to a variable number of visual tokens proportional to its
resolution, with the count controllable through `min_pixels` / `max_pixels`. For a long
document that is the difference between a workable system and an impossible one — see
§2, where the arithmetic is the whole argument.

**Grounding is native.** The model was trained to emit bounding boxes and point
references, not merely to describe. A system whose premise is "every value names where
it came from" is asking the model for something it already knows how to do, rather than
teaching it a new output convention from scratch.

**7B is the size that fits the deployment.** A 7B model in bf16 is ~15 GB of weights,
which serves on a single `ml.g5.2xlarge` (A10G, 24 GB) with room for KV cache at an 8K
context. A 72B model would want four times the instance for a task where the smaller
model, once adapted, was already producing 0.986 schema-valid output. The interesting
constraint was never model capacity — it was context budget.

---

## 2. The visual token budget, which is the real constraint

This is the arithmetic that shapes every other decision.

Qwen2.5-VL's vision encoder uses a 14×14 patch with 2×2 spatial merging, so one visual
token covers a **28×28 pixel** region. A page rendered at 200 DPI on US Letter is about
1700×2200 px:

```
(1700 / 28) × (2200 / 28)  ≈  61 × 79  ≈  4,800 visual tokens per page
```

At four pages per group that is **~19,200 visual tokens before a single word of the
schema, the layout text, or the carry-over state**. Against a practical 32K context,
the prompt no longer fits — and every token spent on page pixels is a token not spent
on the instructions that make the output usable.

`max_pixels` is the lever:

| `max_pixels` | Tokens/page | 4-page group | Fits an 8K context? |
|---|---:|---:|---|
| `1280 × 28 × 28` (default here) | ~1,280 | ~5,120 | Yes, with ~3K for text |
| `1024 × 28 × 28` | ~1,024 | ~4,096 | Yes, comfortably |
| `2048 × 28 × 28` | ~2,048 | ~8,192 | No |
| unbounded at 200 DPI | ~4,800 | ~19,200 | No |

Capping resolution loses fine visual detail. **That loss is what the OCR/layout signal
is there to cover.** The image carries structure — table geometry, stamps, which column
a number sits in — and the layout blocks carry exact characters and coordinates. Fusing
them means the resolution cap costs structure the model can still infer, not characters
it can no longer read. A pure-vision pipeline would have to choose between fitting the
document and reading it.

This is also why `max_chars_per_page` (6,000 by default) exists as a separate bound.
Visual tokens and text tokens compete for the same context, so both need a ceiling.

---

## 3. Where the adapter goes

```python
LoraConfig(
    r=16,
    alpha=32,
    dropout=0.05,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"),
    bias="none",
)
```

Attention projections **and** MLP projections, in the language model only. Nothing in
the vision tower.

### Why the vision tower is frozen

Two reasons, and the second is the stronger one.

The mechanical reason: at 7B, the vision encoder is a small fraction of the parameters
but a large fraction of the activation memory during backward. Freezing it and enabling
gradient checkpointing is what makes an 8K-sequence batch fit on a 24 GB card at all.

The substantive reason: **the failure modes were never visual.** The base model reads
invoices fine. What it does wrong before adaptation is emit a field the schema does not
declare, restart a table at a page boundary, cite a block id that does not exist, and
re-state a value the carry-over already settled. Every one of those is a
language-modelling failure about output convention and context use. Adapting the vision
encoder spends parameters where the bottleneck is not.

### Parameter count

With r=16 on seven projection types across the language layers, trainable parameters
land around **20M against 7B** — roughly 0.3%. The adapter serialises to tens of
megabytes, which is what makes one shared base checkpoint across every document schema
practical: a new schema is a new adapter, not a new 15 GB artefact.

### Why r=16

| r | Observed |
|---:|---|
| 8 | Underfits table continuation — the model still restarted tables at boundaries |
| **16** | Continuation learned; the working default |
| 32 | No gain worth the extra memory and the longer step time |

Table continuation is the discriminating behaviour. Header fields are easy enough that
r=8 handles them; deciding whether a row at the top of page 8 continues page 7's table
or starts a new one is where capacity started to matter.

---

## 4. Loss on the completion only

The prompt for one page group contains the full schema, the block-addressed layout text
for up to four pages, and the carry-over. That is the large majority of the sequence,
and **the model is handed all of it at inference time**. Training it to reproduce that
text spends most of the gradient learning to echo the input.

`CompletionOnlyCollator` finds the assistant-turn marker in each tokenised example and
sets every label before it to `-100`:

```python
cut = last_index_of(input_ids, tokenizer("<|im_start|>assistant"))
labels = [-100] * cut + labels[cut:]
```

On a typical 8K example with a ~600-token target, this moves the loss from being
computed over ~8,000 positions to ~600 — so essentially all of the useful signal, and
none of the echo.

The marker search runs backwards, because the assistant marker can legitimately appear
earlier inside quoted layout text; the last occurrence is the real turn boundary. If no
marker is found the collator logs and falls back to full-sequence loss rather than
silently training on nothing.

---

## 5. Memory and throughput

Single `ml.g5.2xlarge` (1× A10G, 24 GB), bf16, sequence length 8,192:

| | |
|---|---|
| Base weights (bf16) | ~15 GB |
| LoRA adapter + optimiser state | ~0.5 GB |
| Activations with gradient checkpointing | ~4 GB at batch 1 |
| Headroom | ~4 GB |

Which is why `per_device_batch_size=1` with `gradient_accumulation_steps=16`: an
effective batch of 16 is reached without exceeding memory. Page-group prompts are long;
the batch size is set by sequence length, not by preference.

Gradient checkpointing costs roughly 30% throughput and buys the activation memory that
makes the 8K sequence possible at all. Without it, sequence length would have to drop —
and truncating a page-group prompt drops the end of the last page, which is exactly
where a continued table lives.

Spot instances are safe here because the trainer checkpoints to
`checkpoint_s3_uri`; an interruption resumes rather than restarting.

---

## 6. What the model is actually taught

Four behaviours, each demonstrated by construction in the training data rather than
merely instructed in the prompt (see
[`training/dataset.py`](../src/throughline/training/dataset.py)):

| Behaviour | How the data teaches it |
|---|---|
| **Emit only the declared schema** | Targets are built from the schema; undeclared keys never appear in one |
| **Cite the block you read** | Every target field carries the gold `block_id` for the group's pages |
| **Announce a continuing table** | `open_tables` is populated when gold rows exist beyond the group's last page |
| **Do not repeat what is settled** | If group 0 settles `invoice_number`, group 1's target omits it |

The last one is the reason carry-over is *simulated rather than idealised*. Group *n*'s
prompt holds only what groups 0..*n*−1 could actually have produced, so the model learns
to work from partial information — which is the only kind it will ever have.

---

## 7. Decoding

```python
GenerationConfig(
    max_new_tokens=4096,
    do_sample=False,        # greedy
    no_repeat_ngram_size=0, # opt-in; 20-40 for long tabular output
)
```

**Greedy, always.** Extraction has one correct answer. Sampling adds variance to a task
where variance is purely a cost, and it makes a cached result and a fresh result differ
for no reason.

**The repetition guard is opt-in.** Long-horizon autoregressive decoding over a table is
where models fall into repetition loops — the same row emitted forty times until the
token budget runs out. A no-repeat n-gram constraint in the 20–40 range breaks that,
but it is off by default because it can also suppress *legitimate* repetition: two
identical line items on a real invoice are two rows, not a loop. It is a per-schema
decision, which is why it is a config field.

This mirrors the technique in
[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) and DeepSeek-OCR, which use an
n-gram logit processor with a sliding window for exactly this failure on long-horizon
document parsing.

**Schema-constrained where the stack allows it.** `build_grammar()` compiles the schema
to JSON Schema for `guided_json` (vLLM), `json_schema` (SGLang), or Outlines. Where the
serving stack cannot do grammars, the defensive parser in
[`decoding/constrained.py`](../src/throughline/decoding/constrained.py) takes over — and
records every repair it had to make, so how often the soft path is doing work is a
measurable thing rather than a guess.

---

## 8. Serving

| | Real-time endpoint | Async endpoint |
|---|---|---|
| Latency | Seconds, per page group | Minutes, per document |
| Payload | Bounded by the real-time limit | Large — four page images fit comfortably |
| Idle cost | A warm instance | **Scales to zero** |
| Use | Interactive extraction | Corpus processing in batches |

For a corpus that arrives in weekly drops rather than continuously, the async endpoint
is the difference between paying for a GPU all week and paying for the hours it runs.
Both are behind the same `VLMBackend` protocol, so switching is a config change, not a
code change.

---

## 9. What would be next

Honest gaps, in the order they would be worth closing:

1. **Quantised serving.** AWQ or GPTQ at 4-bit would fit the 7B on a `g5.xlarge`,
   roughly halving inference cost. Needs an accuracy check against the bf16 champion —
   exactly what the promotion gate is for.
2. **A cross-encoder reranker for relevant-page retrieval.** BM25 plus positional priors
   is strong for schema fields, whose anchors are short literal strings. It is weaker for
   semantically-phrased fields ("the clause limiting liability"), where a small
   cross-encoder over page text would help.
3. **Per-field confidence calibration.** The model's self-reported confidence is
   uncalibrated. Fitting a per-field mapping from confidence to observed accuracy would
   make the early-exit threshold mean something absolute rather than relative.
4. **Adapting the vision tower for degraded scans.** Frozen is right for clean digital
   documents. It would stop being right for faxes and photographs, where the failures
   genuinely are visual.
