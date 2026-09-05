"""Command-line interface.

::

    throughline extract   examples/documents/*.json --schema invoice
    throughline evaluate  examples/corpus --schema invoice --profile balanced
    throughline sweep     examples/corpus --schema invoice
    throughline build-dataset examples/corpus --schema invoice --out data/train.jsonl
    throughline inspect   examples/documents/invoice_0001.json --schema invoice
    throughline registry  --promote lora-r16-v3 --strict
    throughline pipeline  --schema invoice --bucket my-bucket
    throughline schemas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from throughline import __version__

LOGGER = logging.getLogger("throughline")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_documents(paths: Sequence[str]) -> list[Any]:
    from throughline.ingest.ocr import JsonFixtureProvider, PyMuPdfProvider

    documents = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"No such document: {path}")
        provider = PyMuPdfProvider() if path.suffix.lower() == ".pdf" else JsonFixtureProvider()
        documents.append(provider.extract(path))
    return documents


def _resolve_config(args: argparse.Namespace) -> Any:
    from throughline.config import RunConfig, profile

    config = RunConfig.from_json_file(args.config) if args.config else profile(args.profile)
    config.schema = args.schema
    if getattr(args, "backend", None):
        config.backend.kind = args.backend
    if getattr(args, "endpoint", None):
        config.backend.endpoint_name = args.endpoint
    if getattr(args, "adapter", None):
        config.backend.adapter_path = args.adapter
    if getattr(args, "no_cache", False):
        config.cache.enabled = False
    return config


# ── commands ──────────────────────────────────────────────────────────
def cmd_extract(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    pipeline = config.build_pipeline()
    documents = _load_documents(args.documents)

    outputs = []
    for document in documents:
        result = pipeline.run(document)
        outputs.append(result.to_dict())
        print(result.summary(), file=sys.stderr)

        if not args.quiet:
            print(json.dumps(result.record, indent=2, default=str))
        if args.show_evidence:
            print("\nEvidence:", file=sys.stderr)
            for key, refs in result.state.evidence_map().items():
                cites = ", ".join(f"p{r['page_number']}:{r.get('block_id')}" for r in refs[:3])
                print(f"  {key}: {cites}", file=sys.stderr)

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(outputs if len(outputs) > 1 else outputs[0], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {destination}", file=sys.stderr)

    return 0 if all(item["validation"]["is_valid"] for item in outputs) else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    from throughline.evaluation.harness import EvaluationConfig, evaluate, load_corpus
    from throughline.schema import registry

    config = _resolve_config(args)
    corpus = load_corpus(args.corpus)
    if not corpus:
        print(f"No labelled documents found in {args.corpus}", file=sys.stderr)
        return 1

    run = evaluate(
        config.build_pipeline(),
        corpus,
        registry.get(args.schema),
        EvaluationConfig(
            run_name=args.run_name or f"{args.schema}-{config.profile}",
            params=config.flat_params(),
            tags={"schema": args.schema, "profile": config.profile},
            log_to_mlflow=not args.no_mlflow,
            artifacts_dir=args.artifacts,
        ),
    )

    print(run.summary())
    print()
    print(run.report.table())
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Evaluate every profile on the same corpus and print the comparison."""
    from throughline.config import PROFILES
    from throughline.evaluation.harness import (
        EvaluationConfig,
        compare,
        evaluate,
        load_corpus,
    )
    from throughline.schema import registry

    corpus = load_corpus(args.corpus)
    if not corpus:
        print(f"No labelled documents found in {args.corpus}", file=sys.stderr)
        return 1

    runs = []
    for name in args.profiles or list(PROFILES):
        config = PROFILES[name]
        config.schema = args.schema
        runs.append(
            evaluate(
                config.build_pipeline(),
                corpus,
                registry.get(args.schema),
                EvaluationConfig(
                    run_name=f"{args.schema}-{name}",
                    params=config.flat_params(),
                    tags={"schema": args.schema, "profile": name},
                    log_to_mlflow=not args.no_mlflow,
                ),
            )
        )

    print(compare(runs))
    if args.out:
        Path(args.out).write_text(
            json.dumps([run.to_dict() for run in runs], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from throughline.evaluation.harness import load_corpus
    from throughline.grouping.page_groups import GroupingConfig
    from throughline.schema import registry
    from throughline.training.dataset import build_corpus, split, write_jsonl

    corpus = load_corpus(args.corpus)
    examples = build_corpus(
        corpus,
        registry.get(args.schema),
        grouping=GroupingConfig(max_pages=args.max_pages, overlap=args.overlap),
        include_empty_ratio=args.include_empty_ratio,
    )

    train, validation = split(examples, train_fraction=args.train_fraction)
    out = Path(args.out)
    written = write_jsonl(train, out)
    print(f"train: {written} examples -> {out}", file=sys.stderr)

    if validation:
        validation_path = out.with_name(out.stem + "_validation" + out.suffix)
        write_jsonl(validation, validation_path)
        print(f"validation: {len(validation)} examples -> {validation_path}", file=sys.stderr)

    tokens = sum(example.approximate_tokens() for example in examples)
    print(f"~{tokens / 1e6:.2f}M tokens across {len(examples)} page-group sets", file=sys.stderr)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show how a document partitions and which pages the retriever favours."""
    from throughline.grouping.page_groups import GroupingConfig, partition, summarise
    from throughline.retrieval.relevant_pages import RelevantPageRetriever
    from throughline.schema import registry

    schema = registry.get(args.schema)
    document = _load_documents([args.document])[0]

    print(f"document: {document.document_id} · {document.page_count} pages")
    print(f"schema:   {schema.name} v{schema.version} · {len(schema.all_keys)} keys")
    print()

    groups = partition(document, GroupingConfig(max_pages=args.max_pages, overlap=args.overlap), schema)
    print(f"page groups ({len(groups)}):")
    print(summarise(groups))
    print()

    retriever = RelevantPageRetriever(document, schema)
    print("relevant pages per key:")
    for key in schema.all_keys:
        print(f"  {retriever.explain(key)}")
    return 0


def cmd_schemas(args: argparse.Namespace) -> int:
    from throughline.schema import registry

    for name in registry.available():
        schema = registry.get(name)
        required = len(schema.required_keys)
        print(
            f"{name:<20} v{schema.version:<8} "
            f"{len(schema.fields)} fields, {len(schema.tables)} tables, {required} required"
        )
        if args.verbose:
            print(f"  {schema.description}")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    from throughline.caching.store import PromptCache

    cache = PromptCache(cache_dir=args.dir)
    if args.clear:
        removed = cache.clear()
        print(f"removed {removed} entries from {args.dir}")
        return 0
    entries, size = cache.size()
    print(f"{entries} entries · {size / 1e6:.1f} MB in {args.dir}")
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    """Inspect the model registry, or promote / roll back a model."""
    from throughline.training.registry import STRICT, ModelRegistry, PromotionGate

    registry = ModelRegistry(path=args.path)

    if args.promote:
        gate = STRICT if args.strict else PromotionGate()
        decision = registry.promote(args.promote, gate, force=args.force)
        print(decision.report())
        return 0 if decision.promote else 1

    if args.rollback:
        restored = registry.rollback(schema=args.schema or None)
        if restored is None:
            print("nothing to roll back to", file=sys.stderr)
            return 1
        print(f"rolled back to {restored.model_id}")
        return 0

    print(registry.table())
    champion = registry.champion(args.schema or None)
    if champion:
        print(f"\nchampion: {champion.model_id} · {champion.adapter_uri}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Render or export the retraining DAG."""
    from throughline.training.pipeline import (
        PipelineConfig,
        export_definition,
        render_plan,
        validate_plan,
    )

    config = PipelineConfig(
        schema=args.schema,
        bucket=args.bucket or "BUCKET",
        role_arn=args.role or "",
        endpoint_name=args.endpoint or "throughline-qwen25vl",
    )

    problems = validate_plan(config)
    if problems:
        print("pipeline plan is invalid:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if args.out:
        path = export_definition(config, args.out)
        print(f"wrote {path}", file=sys.stderr)
    print(render_plan(config))
    return 0

# ── parser ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="throughline",
        description="Cross-page state for long-document multimodal extraction.",
    )
    parser.add_argument("--version", action="version", version=f"throughline {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--schema", default="invoice", help="registered schema name")
        sub.add_argument(
            "--profile", default="balanced", choices=["accuracy", "balanced", "fast"]
        )
        sub.add_argument("--config", help="RunConfig JSON file (overrides --profile)")
        sub.add_argument(
            "--backend", choices=["rule-based", "qwen-vl", "sagemaker", "sagemaker-async"]
        )
        sub.add_argument("--endpoint", help="SageMaker endpoint name")
        sub.add_argument("--adapter", help="LoRA adapter directory")
        sub.add_argument("--no-cache", action="store_true", help="bypass the prompt cache")

    extract = subparsers.add_parser("extract", help="extract one or more documents")
    extract.add_argument("documents", nargs="+")
    extract.add_argument("--out", help="write results to this JSON file")
    extract.add_argument("--show-evidence", action="store_true")
    extract.add_argument("--quiet", action="store_true", help="suppress the record on stdout")
    add_common(extract)
    extract.set_defaults(func=cmd_extract)

    evaluate_cmd = subparsers.add_parser("evaluate", help="score against a labelled corpus")
    evaluate_cmd.add_argument("corpus")
    evaluate_cmd.add_argument("--run-name")
    evaluate_cmd.add_argument("--artifacts", help="directory for per-document results")
    evaluate_cmd.add_argument("--no-mlflow", action="store_true")
    add_common(evaluate_cmd)
    evaluate_cmd.set_defaults(func=cmd_evaluate)

    sweep = subparsers.add_parser("sweep", help="compare every profile on one corpus")
    sweep.add_argument("corpus")
    sweep.add_argument("--schema", default="invoice")
    sweep.add_argument("--profiles", nargs="*", choices=["accuracy", "balanced", "fast"])
    sweep.add_argument("--out", help="write the comparison to this JSON file")
    sweep.add_argument("--no-mlflow", action="store_true")
    sweep.set_defaults(func=cmd_sweep)

    dataset = subparsers.add_parser("build-dataset", help="build page-group training data")
    dataset.add_argument("corpus")
    dataset.add_argument("--schema", default="invoice")
    dataset.add_argument("--out", default="data/train.jsonl")
    dataset.add_argument("--max-pages", type=int, default=4)
    dataset.add_argument("--overlap", type=int, default=1)
    dataset.add_argument("--train-fraction", type=float, default=0.9)
    dataset.add_argument("--include-empty-ratio", type=float, default=0.1)
    dataset.set_defaults(func=cmd_build_dataset)

    inspect = subparsers.add_parser("inspect", help="show grouping and page relevance")
    inspect.add_argument("document")
    inspect.add_argument("--schema", default="invoice")
    inspect.add_argument("--max-pages", type=int, default=4)
    inspect.add_argument("--overlap", type=int, default=1)
    inspect.set_defaults(func=cmd_inspect)

    schemas = subparsers.add_parser("schemas", help="list registered schemas")
    schemas.add_argument("--verbose", action="store_true")
    schemas.set_defaults(func=cmd_schemas)

    registry = subparsers.add_parser("registry", help="model registry: list, promote, roll back")
    registry.add_argument("--path", default="results/model_registry.json")
    registry.add_argument("--schema", default="", help="filter to one schema")
    registry.add_argument("--promote", metavar="MODEL_ID", help="run the gate and promote")
    registry.add_argument("--strict", action="store_true", help="use the strict gate profile")
    registry.add_argument("--force", action="store_true", help="promote over a failed gate")
    registry.add_argument("--rollback", action="store_true", help="restore the previous champion")
    registry.set_defaults(func=cmd_registry)

    pipeline = subparsers.add_parser("pipeline", help="render the retraining DAG")
    pipeline.add_argument("--schema", default="invoice")
    pipeline.add_argument("--bucket", help="S3 bucket for artefacts")
    pipeline.add_argument("--role", help="SageMaker execution role ARN")
    pipeline.add_argument("--endpoint", help="endpoint updated on promotion")
    pipeline.add_argument("--out", help="write the definition JSON here")
    pipeline.set_defaults(func=cmd_pipeline)

    cache = subparsers.add_parser("cache", help="inspect or clear the prompt cache")
    cache.add_argument("--dir", default=".cache/prompts")
    cache.add_argument("--clear", action="store_true")
    cache.set_defaults(func=cmd_cache)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, KeyError, ValueError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
