.PHONY: help install test lint fmt results examples demo sweep clean

help:
	@echo "Throughline — cross-page state for long-document multimodal extraction"
	@echo ""
	@echo "  make install    editable install with dev extras"
	@echo "  make test       run the test suite"
	@echo "  make lint       ruff check"
	@echo "  make fmt        ruff check --fix"
	@echo "  make examples   regenerate the synthetic corpus"
	@echo "  make results    regenerate results/tables and results/figures"
	@echo "  make sweep      compare accuracy / balanced / fast profiles"
	@echo "  make demo       inspect + extract + sweep, end to end"
	@echo "  make clean      remove caches and build artefacts"

install:
	pip install -e ".[dev,viz]"

test:
	pytest -q

lint:
	ruff check src tools tests

fmt:
	ruff check --fix src tools tests

examples:
	python tools/make_examples.py

results:
	python tools/make_results.py
	python tools/make_figures.py

sweep:
	throughline sweep examples/corpus            --schema invoice           --no-mlflow
	throughline sweep examples/corpus_agreements --schema service_agreement --no-mlflow

demo:
	@echo "── how a long agreement partitions ─────────────────────────"
	throughline inspect examples/documents/agreement_0001.json --schema service_agreement
	@echo ""
	@echo "── extraction with verified citations ──────────────────────"
	throughline extract examples/documents/invoice_0001.json --schema invoice \
	  --show-evidence --quiet
	@echo ""
	@echo "── accuracy / coverage trade-off ───────────────────────────"
	throughline sweep examples/corpus_agreements --schema service_agreement --no-mlflow

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .cache build dist *.egg-info
	rm -rf src/*.egg-info
