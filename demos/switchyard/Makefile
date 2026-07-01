ROOT := ../..
DD_PYTHON := $(ROOT)/.venv/bin/python
DD_PYTEST := $(ROOT)/.venv/bin/pytest
SWITCHYARD_PYTHON ?= python3
OUTPUT_DIR ?= $(ROOT)/.scratch/switchyard-experiments
SUMMARY_PATH ?= $(CURDIR)/results.json
PDF_PATH ?= $(ROOT)/output/pdf/switchyard-data-designer-experiments.pdf
RUN_ARGS ?=

.PHONY: serve run report test

serve:
	@test -n "$(NVIDIA_INFERENCE_BASE_URL)" || (echo "NVIDIA_INFERENCE_BASE_URL is required" && false)
	@test -n "$(NVIDIA_INFERENCE_API_KEY)" || (echo "NVIDIA_INFERENCE_API_KEY is required" && false)
	$(SWITCHYARD_PYTHON) serve.py

run:
	@test -n "$(SWITCHYARD_BASE_URL)" || (echo "SWITCHYARD_BASE_URL is required" && false)
	$(DD_PYTHON) run_experiments.py --output-dir $(OUTPUT_DIR) --summary-path $(SUMMARY_PATH) $(RUN_ARGS)

report:
	@mkdir -p $(dir $(PDF_PATH))
	typst compile report.typ $(PDF_PATH)

test:
	$(DD_PYTEST) tests
