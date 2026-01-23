REPO_PATH := $(shell pwd)

define install-pre-commit-hooks
	@if [ ! -f $(REPO_PATH)/.git/hooks/pre-commit ]; then \
		echo "🪝 Installing pre-commit hooks..."; \
		uv run pre-commit install; \
	else \
		echo "👍 Pre-commit hooks already installed"; \
	fi
endef

help:
	@echo ""
	@echo "🚀 DataDesigner Makefile Commands"
	@echo "═════════════════════════════════════════════════════════════"
	@echo ""
	@echo "📦 Installation (uv workspace - all packages in editable mode):"
	@echo "  install                   - Install all packages (config → engine → interface)"
	@echo "  install-dev               - Install all packages + dev tools (pytest, etc.)"
	@echo "  install-dev-notebooks     - Install all packages + dev + notebook tools"
	@echo ""
	@echo "🧪 Testing (all packages):"
	@echo "  test                      - Run all unit tests"
	@echo "  coverage                  - Run tests with coverage report"
	@echo "  test-e2e                  - Run e2e plugin tests"
	@echo "  test-run-tutorials        - Run tutorial notebooks as e2e tests"
	@echo "  test-run-recipes          - Run recipe scripts as e2e tests"
	@echo "  test-run-all-examples     - Run all tutorials and recipes as e2e tests"
	@echo ""
	@echo "✨ Code Quality (all packages):"
	@echo "  format                    - Format all code with ruff"
	@echo "  format-check              - Check code formatting without making changes"
	@echo "  lint                      - Lint all code with ruff"
	@echo "  lint-fix                  - Fix linting issues automatically"
	@echo "  build                     - Build all package wheels"
	@echo ""
	@echo "🔍 Combined Checks:"
	@echo "  check-all                 - Run all checks (format-check + lint)"
	@echo "  check-all-fix             - Run all checks with autofix (format + lint-fix)"
	@echo ""
	@echo "🛠️  Utilities:"
	@echo "  clean                     - Remove coverage reports, cache files, and dist"
	@echo "  clean-dist                - Remove dist directories from all packages"
	@echo "  verify-imports            - Verify all package imports work"
	@echo "  show-versions             - Show versions of all packages"
	@echo "  convert-execute-notebooks - Convert notebooks from .py to .ipynb using jupytext"
	@echo "  generate-colab-notebooks  - Generate Colab-compatible notebooks"
	@echo "  serve-docs-locally        - Serve documentation locally"
	@echo "  check-license-headers     - Check if all files have license headers"
	@echo "  update-license-headers    - Add license headers to all files"
	@echo ""
	@echo "⚡ Performance:"
	@echo "  perf-import               - Profile import time and show summary"
	@echo "  perf-import CLEAN=1       - Clean cache, then profile import time"
	@echo "  perf-import NOFILE=1      - Profile without writing to file (for CI)"
	@echo ""
	@echo "📦 Per-Package Commands (use suffix: -config, -engine, -interface):"
	@echo "  test-<pkg>                - Run tests for a specific package"
	@echo "  lint-<pkg>                - Lint a specific package"
	@echo "  lint-fix-<pkg>            - Fix lint issues in a specific package"
	@echo "  format-<pkg>              - Format a specific package"
	@echo "  check-<pkg>               - Check format + lint for a specific package"
	@echo "  build-<pkg>               - Build wheel for a specific package"
	@echo "  coverage-<pkg>            - Run tests with coverage for a specific package"
	@echo ""
	@echo "═════════════════════════════════════════════════════════════"
	@echo "💡 Tip: Run 'make <command>' to execute any command above"
	@echo ""

clean-pycache:
	@echo "🧹 Cleaning up Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cache cleaned!"

clean-dist:
	@echo "🧹 Cleaning dist directories..."
	rm -rf packages/data-designer-config/dist
	rm -rf packages/data-designer-engine/dist
	rm -rf packages/data-designer/dist
	@echo "✅ Dist directories cleaned!"

clean: clean-pycache clean-dist
	@echo "🧹 Cleaning up coverage reports and test cache..."
	rm -rf htmlcov .coverage .pytest_cache
	rm -f packages/*/src/data_designer/*/_version.py
	@echo "✅ Cleaned!"

coverage:
	@echo "📊 Running tests with coverage analysis (all packages)..."
	uv run --group dev pytest \
		packages/data-designer-config/tests \
		packages/data-designer-engine/tests \
		packages/data-designer/tests \
		--cov=data_designer \
		--cov-report=term-missing \
		--cov-report=html
	@echo "✅ Coverage report generated in htmlcov/index.html"

check-all: format-check lint
	@echo "✅ All checks complete!"

check-all-fix: format lint-fix
	@echo "✅ All checks with autofix complete!"

format:
	@echo "📐 Formatting code with ruff..."
	uv run ruff format packages/ scripts/ tests_e2e/ --exclude '**/_version.py'
	@echo "✅ Formatting complete!"

format-check:
	@echo "📐 Checking code formatting with ruff..."
	uv run ruff format --check packages/ scripts/ tests_e2e/ --exclude '**/_version.py'
	@echo "✅ Formatting check complete! Run 'make format' to auto-fix issues."

lint:
	@echo "🔍 Linting code with ruff..."
	uv run ruff check --output-format=full packages/ scripts/ tests_e2e/ --exclude '**/_version.py'
	@echo "✅ Linting complete! Run 'make lint-fix' to auto-fix issues."

lint-fix:
	@echo "🔍 Fixing linting issues with ruff..."
	uv run ruff check --fix packages/ scripts/ tests_e2e/ --exclude '**/_version.py'
	@echo "✅ Linting with autofix complete!"

test: test-config test-engine test-interface
	@echo "✅ All package tests complete!"

test-e2e:
	@echo "🧹 Cleaning e2e test environment..."
	rm -rf tests_e2e/uv.lock tests_e2e/.pycache tests_e2e/.venv
	@echo "🧪 Running e2e tests..."
	uv run --no-cache --refresh --directory tests_e2e pytest -s

test-run-tutorials:
	@echo "🧪 Running tutorials as e2e tests..."
	@TUTORIAL_WORKDIR=$$(mktemp -d); \
	trap "rm -rf $$TUTORIAL_WORKDIR" EXIT; \
	for f in docs/notebook_source/*.py; do \
		echo "  📓 Running $$f..."; \
		(cd "$$TUTORIAL_WORKDIR" && uv run --project "$(REPO_PATH)" --group notebooks python "$(REPO_PATH)/$$f") || exit 1; \
	done; \
	echo "🧹 Cleaning up tutorial artifacts..."; \
	rm -rf "$$TUTORIAL_WORKDIR"; \
	echo "✅ All tutorials completed successfully!"

test-run-recipes:
	@echo "🧪 Running recipes as e2e tests..."
	@RECIPE_WORKDIR=$$(mktemp -d); \
	trap "rm -rf $$RECIPE_WORKDIR" EXIT; \
	for f in docs/assets/recipes/**/*.py; do \
		echo "  📜 Running $$f..."; \
		(cd "$$RECIPE_WORKDIR" && uv run --project "$(REPO_PATH)" --group notebooks python "$(REPO_PATH)/$$f" --model-alias nvidia-text --artifact-path "$$RECIPE_WORKDIR" --num-records 5) || exit 1; \
	done; \
	echo "🧹 Cleaning up recipe artifacts..."; \
	rm -rf "$$RECIPE_WORKDIR"; \
	echo "✅ All recipes completed successfully!"

test-run-all-examples: test-run-tutorials test-run-recipes
	@echo "✅ All examples (tutorials + recipes) completed successfully!"

convert-execute-notebooks:
	@echo "📓 Converting Python tutorials to notebooks and executing..."
	@mkdir -p docs/notebooks
	cp docs/notebook_source/_README.md docs/notebooks/README.md
	cp docs/notebook_source/_pyproject.toml docs/notebooks/pyproject.toml
	uv run --all-packages --group notebooks --group docs jupytext --to ipynb --execute docs/notebook_source/*.py
	mv docs/notebook_source/*.ipynb docs/notebooks/
	rm -r docs/notebook_source/artifacts
	rm docs/notebook_source/*.csv
	@echo "✅ Notebooks created in docs/notebooks/"

generate-colab-notebooks:
	@echo "📓 Generating Colab-compatible notebooks..."
	uv run --group docs python docs/scripts/generate_colab_notebooks.py
	@echo "✅ Colab notebooks created in docs/colab_notebooks/"

serve-docs-locally:
	@echo "📝 Building and serving docs..."
	uv sync --group docs
	uv run mkdocs serve --livereload

check-license-headers:
	@echo "🔍 Checking license headers in all files..."
	uv run python $(REPO_PATH)/scripts/update_license_headers.py --check

update-license-headers:
	@echo "🔍 Updating license headers in all files..."
	uv run python $(REPO_PATH)/scripts/update_license_headers.py

verify-imports:
	@echo "🔍 Verifying package imports..."
	uv run python -c "from data_designer.config.config_builder import DataDesignerConfigBuilder; print('  ✓ config')"
	uv run python -c "from data_designer.engine.compiler import compile_data_designer_config; print('  ✓ engine')"
	uv run python -c "from data_designer.interface.data_designer import DataDesigner; print('  ✓ interface')"
	@echo "✅ All imports verified!"

show-versions:
	@echo "📦 Package versions:"
	@uv run python -c "from data_designer.config._version import __version__; print(f'  data-designer-config: {__version__}')" 2>/dev/null || echo "  data-designer-config: (not installed)"
	@uv run python -c "from data_designer.engine._version import __version__; print(f'  data-designer-engine: {__version__}')" 2>/dev/null || echo "  data-designer-engine: (not installed)"
	@uv run python -c "from data_designer.interface._version import __version__; print(f'  data-designer:        {__version__}')" 2>/dev/null || echo "  data-designer: (not installed)"

install:
	@echo "📦 Installing DataDesigner workspace (all packages in editable mode)..."
	@echo "   Packages: data-designer-config → data-designer-engine → data-designer"
	uv sync --all-packages
	@echo "✅ Installation complete!"
	@echo ""
	@echo "💡 Run 'make verify-imports' to verify all packages are working"

install-dev:
	@echo "📦 Installing DataDesigner workspace in development mode..."
	@echo "   Packages: data-designer-config → data-designer-engine → data-designer"
	@echo "   Groups: dev (pytest, coverage, etc.)"
	uv sync --all-packages --group dev
	$(call install-pre-commit-hooks)
	@echo ""
	@echo "✅ All packages installed in development mode!"
	@echo ""
	@echo "📁 Workspace structure:"
	@echo "   packages/data-designer-config/   - Configuration layer (lightweight)"
	@echo "   packages/data-designer-engine/   - Generation engine (heavy deps)"
	@echo "   packages/data-designer/          - Full package with CLI"
	@echo ""
	@echo "💡 Next steps:"
	@echo "   make verify-imports     - Verify all packages are working"
	@echo "   make test               - Run all tests across packages"
	@echo "   make test-<pkg>         - Run tests for specific package (config, engine, interface)"
	@echo "   make lint               - Lint all code"
	@echo "   make build              - Build all package wheels"

install-dev-notebooks:
	@echo "📦 Installing DataDesigner workspace with notebook dependencies..."
	@echo "   Packages: data-designer-config → data-designer-engine → data-designer"
	@echo "   Groups: dev + notebooks (Jupyter, jupytext, etc.)"
	uv sync --all-packages --group dev --group notebooks
	$(call install-pre-commit-hooks)
	@echo "✅ Dev + notebooks installation complete!"
	@echo ""
	@echo "💡 Run 'make test-run-tutorials' to test notebook tutorials"

perf-import:
ifdef CLEAN
	@$(MAKE) clean-pycache
endif
	@echo "⚡ Profiling import time for data_designer.config and data_designer.interface.DataDesigner..."
ifdef NOFILE
	@PERF_OUTPUT=$$(uv run python -X importtime -c "import data_designer.config as dd; from data_designer.interface import DataDesigner; DataDesigner(); dd.DataDesignerConfigBuilder()" 2>&1); \
	echo "$$PERF_OUTPUT"; \
	echo ""; \
	echo "Summary:"; \
	echo "$$PERF_OUTPUT" | tail -1 | awk '{printf "  Total: %.3fs\n", $$5/1000000}'; \
	echo ""; \
	echo "💡 Top 10 slowest imports:"; \
	printf "%-12s %-12s %s\n" "Self (s)" "Cumulative (s)" "Module"; \
	printf "%-12s %-12s %s\n" "--------" "--------------" "------"; \
	echo "$$PERF_OUTPUT" | grep "import time:" | sort -rn -k5 | head -10 | awk '{printf "%-12.3f %-12.3f %s", $$3/1000000, $$5/1000000, $$7; for(i=8;i<=NF;i++) printf " %s", $$i; printf "\n"}'
else
	@PERF_FILE="perf_import_$$(date +%Y%m%d_%H%M%S).txt"; \
	uv run python -X importtime -c "import data_designer.config as dd; from data_designer.interface import DataDesigner; DataDesigner(); dd.DataDesignerConfigBuilder()" > "$$PERF_FILE" 2>&1; \
	echo "📊 Import profile saved to $$PERF_FILE"; \
	echo ""; \
	echo "Summary:"; \
	tail -1 "$$PERF_FILE" | awk '{printf "  Total: %.3fs\n", $$5/1000000}'; \
	echo ""; \
	echo "💡 Top 10 slowest imports:"; \
	printf "%-12s %-12s %s\n" "Self (s)" "Cumulative (s)" "Module"; \
	printf "%-12s %-12s %s\n" "--------" "--------------" "------"; \
	grep "import time:" "$$PERF_FILE" | sort -rn -k5 | head -10 | awk '{printf "%-12.3f %-12.3f %s", $$3/1000000, $$5/1000000, $$7; for(i=8;i<=NF;i++) printf " %s", $$i; printf "\n"}'
endif

# Subpackage test commands
test-config:
	@echo "🧪 Testing data-designer-config..."
	uv run --group dev pytest packages/data-designer-config/tests

test-engine:
	@echo "🧪 Testing data-designer-engine..."
	uv run --group dev pytest packages/data-designer-engine/tests

test-interface:
	@echo "🧪 Testing data-designer (interface)..."
	uv run --group dev pytest packages/data-designer/tests

# Subpackage lint commands
lint-config:
	@echo "🔍 Linting data-designer-config..."
	uv run ruff check packages/data-designer-config/src packages/data-designer-config/tests

lint-engine:
	@echo "🔍 Linting data-designer-engine..."
	uv run ruff check packages/data-designer-engine/src packages/data-designer-engine/tests

lint-interface:
	@echo "🔍 Linting data-designer (interface)..."
	uv run ruff check packages/data-designer/src packages/data-designer/tests

# Subpackage format commands
format-config:
	@echo "📐 Formatting data-designer-config..."
	uv run ruff format packages/data-designer-config/src packages/data-designer-config/tests

format-engine:
	@echo "📐 Formatting data-designer-engine..."
	uv run ruff format packages/data-designer-engine/src packages/data-designer-engine/tests

format-interface:
	@echo "📐 Formatting data-designer (interface)..."
	uv run ruff format packages/data-designer/src packages/data-designer/tests

# Subpackage lint-fix commands
lint-fix-config:
	@echo "🔍 Fixing lint issues in data-designer-config..."
	uv run ruff check --fix packages/data-designer-config/src packages/data-designer-config/tests

lint-fix-engine:
	@echo "🔍 Fixing lint issues in data-designer-engine..."
	uv run ruff check --fix packages/data-designer-engine/src packages/data-designer-engine/tests

lint-fix-interface:
	@echo "🔍 Fixing lint issues in data-designer (interface)..."
	uv run ruff check --fix packages/data-designer/src packages/data-designer/tests

# Subpackage check commands (format-check + lint)
check-config:
	@echo "🔍 Checking data-designer-config..."
	uv run ruff format --check packages/data-designer-config/src packages/data-designer-config/tests
	uv run ruff check packages/data-designer-config/src packages/data-designer-config/tests

check-engine:
	@echo "🔍 Checking data-designer-engine..."
	uv run ruff format --check packages/data-designer-engine/src packages/data-designer-engine/tests
	uv run ruff check packages/data-designer-engine/src packages/data-designer-engine/tests

check-interface:
	@echo "🔍 Checking data-designer (interface)..."
	uv run ruff format --check packages/data-designer/src packages/data-designer/tests
	uv run ruff check packages/data-designer/src packages/data-designer/tests

# Subpackage build commands
build-config:
	@echo "🏗️  Building data-designer-config..."
	cd packages/data-designer-config && uv build -o dist

build-engine:
	@echo "🏗️  Building data-designer-engine..."
	cd packages/data-designer-engine && uv build -o dist

build-interface:
	@echo "🏗️  Building data-designer (interface)..."
	cd packages/data-designer && uv build -o dist

build: build-config build-engine build-interface
	@echo "✅ All packages built!"

# Subpackage coverage commands
coverage-config:
	@echo "📊 Running config tests with coverage..."
	uv run --group dev pytest packages/data-designer-config/tests --cov=data_designer.config --cov-report=term-missing --cov-report=html

coverage-engine:
	@echo "📊 Running engine tests with coverage..."
	uv run --group dev pytest packages/data-designer-engine/tests --cov=data_designer.engine --cov-report=term-missing --cov-report=html

coverage-interface:
	@echo "📊 Running interface tests with coverage..."
	uv run --group dev pytest packages/data-designer/tests --cov=data_designer --cov-report=term-missing --cov-report=html

.PHONY: clean clean-pycache clean-dist coverage format format-check lint lint-fix test test-e2e test-run-tutorials test-run-recipes test-run-all-examples check-license-headers update-license-headers check-all check-all-fix install install-dev install-dev-notebooks generate-colab-notebooks perf-import verify-imports show-versions test-config test-engine test-interface lint-config lint-engine lint-interface format-config format-engine format-interface lint-fix-config lint-fix-engine lint-fix-interface check-config check-engine check-interface build build-config build-engine build-interface coverage-config coverage-engine coverage-interface
