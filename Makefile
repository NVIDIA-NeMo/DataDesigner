# ==============================================================================
# VARIABLES AND FUNCTIONS
# ==============================================================================

REPO_PATH := $(shell pwd)

# Package directories
CONFIG_PKG := packages/data-designer-config
ENGINE_PKG := packages/data-designer-engine
INTERFACE_PKG := packages/data-designer

# Package source and test paths
CONFIG_PATHS := $(CONFIG_PKG)/src $(CONFIG_PKG)/tests
ENGINE_PATHS := $(ENGINE_PKG)/src $(ENGINE_PKG)/tests
INTERFACE_PATHS := $(INTERFACE_PKG)/src $(INTERFACE_PKG)/tests
ALL_PKG_PATHS := packages/ scripts/ tests_e2e/

# Test directories
CONFIG_TESTS := $(CONFIG_PKG)/tests
ENGINE_TESTS := $(ENGINE_PKG)/tests
INTERFACE_TESTS := $(INTERFACE_PKG)/tests

define install-pre-commit-hooks
	@if [ ! -f $(REPO_PATH)/.git/hooks/pre-commit ]; then \
		echo "🪝 Installing pre-commit hooks..."; \
		uv run pre-commit install; \
	else \
		echo "👍 Pre-commit hooks already installed"; \
	fi
endef

# ==============================================================================
# HELP
# ==============================================================================

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
	@echo "🔬 Isolated Testing (mirrors CI - uses temp venv):"
	@echo "  test-isolated             - Run all isolated tests (config → engine → interface)"
	@echo "  test-config-isolated      - Test config with ONLY config installed"
	@echo "  test-engine-isolated      - Test engine with ONLY engine+config installed"
	@echo "  test-interface-isolated   - Test interface with full package installed"
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
	@echo "🚀 Publish:"
	@echo "  publish VERSION=X.Y.Z                        - Publish all packages to PyPI"
	@echo "  publish VERSION=X.Y.Z DRY_RUN=1              - Dry run (no tags or uploads)"
	@echo "  publish VERSION=X.Y.Z TEST_PYPI=1            - Publish to TestPyPI"
	@echo "  publish VERSION=X.Y.Z ALLOW_BRANCH=1         - Publish from non-main branch"
	@echo "  publish VERSION=X.Y.Z FORCE_TAG=1            - Overwrite existing git tag"
	@echo ""
	@echo "📦 Per-Package Commands (use suffix: -config, -engine, -interface):"
	@echo "  test-<pkg>                - Run tests for a specific package"
	@echo "  lint-<pkg>                - Lint a specific package"
	@echo "  lint-fix-<pkg>            - Fix lint issues in a specific package"
	@echo "  format-<pkg>              - Format a specific package"
	@echo "  format-check-<pkg>        - Check formatting for a specific package"
	@echo "  check-<pkg>               - Check format + lint for a specific package"
	@echo "  build-<pkg>               - Build wheel for a specific package"
	@echo "  coverage-<pkg>            - Run tests with coverage for a specific package"
	@echo ""
	@echo "═════════════════════════════════════════════════════════════"
	@echo "💡 Tip: Run 'make <command>' to execute any command above"
	@echo ""

# ==============================================================================
# INSTALLATION
# ==============================================================================

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
	@echo "📄 Copying top-level README to data-designer package..."
	cp README.md packages/data-designer/README.md
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

# ==============================================================================
# TESTING
# ==============================================================================

test: test-config test-engine test-interface
	@echo "✅ All package tests complete!"

test-config:
	@echo "🧪 Testing data-designer-config..."
	uv run --group dev pytest $(CONFIG_TESTS)

test-engine:
	@echo "🧪 Testing data-designer-engine..."
	uv run --group dev pytest $(ENGINE_TESTS)

test-interface:
	@echo "🧪 Testing data-designer (interface)..."
	uv run --group dev pytest $(INTERFACE_TESTS)

# ------------------------------------------------------------------------------
# Isolated Testing (mirrors CI behavior)
# Each package is installed independently to verify dependency boundaries
# Uses temporary virtual environments to avoid affecting the main dev environment
# ------------------------------------------------------------------------------

# Test dependencies added via --with since workspace groups aren't available with --package
TEST_DEPS := --with pytest --with pytest-asyncio --with pytest-httpx --with pytest-env

test-isolated:
	@echo "🧪 Running all isolated package tests..."
	@echo ""
	@CONFIG_RESULT=0; ENGINE_RESULT=0; INTERFACE_RESULT=0; \
	$(MAKE) test-config-isolated || CONFIG_RESULT=1; \
	echo ""; \
	$(MAKE) test-engine-isolated || ENGINE_RESULT=1; \
	echo ""; \
	$(MAKE) test-interface-isolated || INTERFACE_RESULT=1; \
	echo ""; \
	echo "═══════════════════════════════════════════════════════════"; \
	echo "📊 Isolated Test Summary:"; \
	if [ $$CONFIG_RESULT -eq 0 ]; then echo "  ✅ Config tests passed"; else echo "  ❌ Config tests FAILED"; fi; \
	if [ $$ENGINE_RESULT -eq 0 ]; then echo "  ✅ Engine tests passed"; else echo "  ❌ Engine tests FAILED"; fi; \
	if [ $$INTERFACE_RESULT -eq 0 ]; then echo "  ✅ Interface tests passed"; else echo "  ❌ Interface tests FAILED"; fi; \
	echo "═══════════════════════════════════════════════════════════"; \
	if [ $$CONFIG_RESULT -ne 0 ] || [ $$ENGINE_RESULT -ne 0 ] || [ $$INTERFACE_RESULT -ne 0 ]; then \
		echo "❌ Some isolated tests FAILED!"; \
		exit 1; \
	fi; \
	echo "✅ All isolated package tests passed!"

test-config-isolated:
	@echo "🧪 Testing data-designer-config in isolation..."
	@ISOLATED_VENV=$$(mktemp -d); \
	trap "rm -rf $$ISOLATED_VENV" EXIT; \
	echo "   Creating isolated environment in $$ISOLATED_VENV..."; \
	echo "   Installing config package only (no engine/interface)..."; \
	UV_PROJECT_ENVIRONMENT="$$ISOLATED_VENV" uv sync --package data-designer-config && \
	UV_PROJECT_ENVIRONMENT="$$ISOLATED_VENV" uv run $(TEST_DEPS) pytest -v $(CONFIG_TESTS) && \
	echo "✅ Config tests passed in isolation!" || \
	{ echo "❌ Config tests FAILED in isolation!"; exit 1; }

test-engine-isolated:
	@echo "🧪 Testing data-designer-engine in isolation..."
	@ISOLATED_VENV=$$(mktemp -d); \
	trap "rm -rf $$ISOLATED_VENV" EXIT; \
	echo "   Creating isolated environment in $$ISOLATED_VENV..."; \
	echo "   Installing engine package only (auto-includes config)..."; \
	UV_PROJECT_ENVIRONMENT="$$ISOLATED_VENV" uv sync --package data-designer-engine && \
	UV_PROJECT_ENVIRONMENT="$$ISOLATED_VENV" uv run $(TEST_DEPS) pytest -v $(ENGINE_TESTS) && \
	echo "✅ Engine tests passed in isolation!" || \
	{ echo "❌ Engine tests FAILED in isolation!"; exit 1; }

test-interface-isolated:
	@echo "🧪 Testing data-designer (interface) in isolation..."
	@echo "📄 Copying top-level README to data-designer package..."
	@cp README.md packages/data-designer/README.md
	@ISOLATED_VENV=$$(mktemp -d); \
	trap "rm -rf $$ISOLATED_VENV" EXIT; \
	echo "   Creating isolated environment in $$ISOLATED_VENV..."; \
	echo "   Installing interface package (auto-includes config + engine)..."; \
	UV_PROJECT_ENVIRONMENT="$$ISOLATED_VENV" uv sync --package data-designer && \
	UV_PROJECT_ENVIRONMENT="$$ISOLATED_VENV" uv run $(TEST_DEPS) pytest -v $(INTERFACE_TESTS) && \
	echo "✅ Interface tests passed in isolation!" || \
	{ echo "❌ Interface tests FAILED in isolation!"; exit 1; }

# Note: coverage runs all tests in a single pytest invocation for combined coverage reporting.
# This is intentionally different from calling coverage-config/engine/interface individually.
coverage:
	@echo "📊 Running tests with coverage analysis (all packages)..."
	uv run --group dev pytest \
		$(CONFIG_TESTS) \
		$(ENGINE_TESTS) \
		$(INTERFACE_TESTS) \
		--cov=data_designer \
		--cov-report=term-missing \
		--cov-report=html
	@echo "✅ Coverage report generated in htmlcov/index.html"

coverage-config:
	@echo "📊 Running config tests with coverage..."
	uv run --group dev pytest $(CONFIG_TESTS) --cov=data_designer --cov-report=term-missing --cov-report=html

coverage-engine:
	@echo "📊 Running engine tests with coverage..."
	uv run --group dev pytest $(ENGINE_TESTS) --cov=data_designer --cov-report=term-missing --cov-report=html

coverage-interface:
	@echo "📊 Running interface tests with coverage..."
	uv run --group dev pytest $(INTERFACE_TESTS) --cov=data_designer --cov-report=term-missing --cov-report=html

test-e2e:
	@echo "📄 Copying top-level README to data-designer package..."
	@cp README.md packages/data-designer/README.md
	@echo "🧹 Cleaning e2e test environment..."
	rm -rf tests_e2e/uv.lock tests_e2e/__pycache__ tests_e2e/.venv
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

# ==============================================================================
# CODE QUALITY - FORMATTING
# ==============================================================================

format: format-config format-engine format-interface
	@echo "📐 Formatting scripts and tests_e2e..."
	uv run ruff format scripts/ tests_e2e/
	@echo "✅ Formatting complete!"

format-check: format-check-config format-check-engine format-check-interface
	@echo "📐 Checking scripts and tests_e2e formatting..."
	uv run ruff format --check scripts/ tests_e2e/
	@echo "✅ Formatting check complete! Run 'make format' to auto-fix issues."

format-config:
	@echo "📐 Formatting data-designer-config..."
	uv run ruff format $(CONFIG_PATHS) --exclude '**/_version.py'

format-engine:
	@echo "📐 Formatting data-designer-engine..."
	uv run ruff format $(ENGINE_PATHS) --exclude '**/_version.py'

format-interface:
	@echo "📐 Formatting data-designer (interface)..."
	uv run ruff format $(INTERFACE_PATHS) --exclude '**/_version.py'

format-check-config:
	@echo "📐 Checking data-designer-config formatting..."
	uv run ruff format --check $(CONFIG_PATHS) --exclude '**/_version.py'

format-check-engine:
	@echo "📐 Checking data-designer-engine formatting..."
	uv run ruff format --check $(ENGINE_PATHS) --exclude '**/_version.py'

format-check-interface:
	@echo "📐 Checking data-designer (interface) formatting..."
	uv run ruff format --check $(INTERFACE_PATHS) --exclude '**/_version.py'

# ==============================================================================
# CODE QUALITY - LINTING
# ==============================================================================

lint: lint-config lint-engine lint-interface
	@echo "🔍 Linting scripts and tests_e2e..."
	uv run ruff check --output-format=full scripts/ tests_e2e/
	@echo "✅ Linting complete! Run 'make lint-fix' to auto-fix issues."

lint-fix: lint-fix-config lint-fix-engine lint-fix-interface
	@echo "🔍 Fixing lint issues in scripts and tests_e2e..."
	uv run ruff check --fix scripts/ tests_e2e/
	@echo "✅ Linting with autofix complete!"

lint-config:
	@echo "🔍 Linting data-designer-config..."
	uv run ruff check --output-format=full $(CONFIG_PATHS) --exclude '**/_version.py'

lint-engine:
	@echo "🔍 Linting data-designer-engine..."
	uv run ruff check --output-format=full $(ENGINE_PATHS) --exclude '**/_version.py'

lint-interface:
	@echo "🔍 Linting data-designer (interface)..."
	uv run ruff check --output-format=full $(INTERFACE_PATHS) --exclude '**/_version.py'

lint-fix-config:
	@echo "🔍 Fixing lint issues in data-designer-config..."
	uv run ruff check --fix $(CONFIG_PATHS) --exclude '**/_version.py'

lint-fix-engine:
	@echo "🔍 Fixing lint issues in data-designer-engine..."
	uv run ruff check --fix $(ENGINE_PATHS) --exclude '**/_version.py'

lint-fix-interface:
	@echo "🔍 Fixing lint issues in data-designer (interface)..."
	uv run ruff check --fix $(INTERFACE_PATHS) --exclude '**/_version.py'

# ==============================================================================
# CODE QUALITY - COMBINED CHECKS
# ==============================================================================

check-all: format-check lint
	@echo "✅ All checks complete!"

check-all-fix: format lint-fix
	@echo "✅ All checks with autofix complete!"

check-config: format-check-config lint-config
	@echo "✅ Checks complete for data-designer-config!"

check-engine: format-check-engine lint-engine
	@echo "✅ Checks complete for data-designer-engine!"

check-interface: format-check-interface lint-interface
	@echo "✅ Checks complete for data-designer (interface)!"

# ==============================================================================
# BUILD
# ==============================================================================

build: build-config build-engine build-interface
	@echo "✅ All packages built!"

build-config:
	@echo "🏗️  Building data-designer-config..."
	cd $(CONFIG_PKG) && uv build -o dist

build-engine:
	@echo "🏗️  Building data-designer-engine..."
	cd $(ENGINE_PKG) && uv build -o dist

build-interface:
	@echo "🏗️  Building data-designer (interface)..."
	@echo "📄 Copying top-level README to data-designer package..."
	cp README.md $(INTERFACE_PKG)/README.md
	cd $(INTERFACE_PKG) && uv build -o dist

# ==============================================================================
# UTILITIES
# ==============================================================================

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

# ==============================================================================
# LICENSE HEADERS
# ==============================================================================

check-license-headers:
	@echo "🔍 Checking license headers in all files..."
	uv run python $(REPO_PATH)/scripts/update_license_headers.py --check

update-license-headers:
	@echo "🔍 Updating license headers in all files..."
	uv run python $(REPO_PATH)/scripts/update_license_headers.py

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

serve-docs-locally:
	@echo "📝 Building and serving docs..."
	uv sync --group docs
	uv run mkdocs serve --livereload

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

# ==============================================================================
# PERFORMANCE
# ==============================================================================

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

# ==============================================================================
# PUBLISH
# ==============================================================================

# Build publish flags based on options
PUBLISH_FLAGS :=
ifdef DRY_RUN
PUBLISH_FLAGS += --dry-run
endif
ifdef TEST_PYPI
PUBLISH_FLAGS += --test-pypi
endif
ifdef ALLOW_BRANCH
PUBLISH_FLAGS += --allow-branch
endif
ifdef FORCE_TAG
PUBLISH_FLAGS += --force-tag
endif

publish:
ifndef VERSION
	$(error VERSION is required. Usage: make publish VERSION=0.3.9rc1 [DRY_RUN=1] [TEST_PYPI=1] [ALLOW_BRANCH=1] [FORCE_TAG=1])
endif
ifdef TEST_PYPI
	@echo "🚀 Publishing version $(VERSION) to TestPyPI..."
else
ifdef DRY_RUN
	@echo "🚀 Running publish dry-run for version $(VERSION)..."
else
	@echo "🚀 Publishing version $(VERSION) to PyPI..."
endif
endif
	$(REPO_PATH)/scripts/publish.sh $(VERSION) $(PUBLISH_FLAGS)

# ==============================================================================
# CLEANUP
# ==============================================================================

clean: clean-pycache clean-dist clean-notebooks clean-test-coverage
	@echo "✅ Cleaned!"

clean-pycache:
	@echo "🧹 Cleaning up Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cache cleaned!"

clean-dist:
	@echo "🧹 Cleaning dist directories..."
	rm -rf $(CONFIG_PKG)/dist
	rm -rf $(ENGINE_PKG)/dist
	rm -rf $(INTERFACE_PKG)/dist
	rm -f packages/*/src/data_designer/*/_version.py
	@echo "✅ Dist directories cleaned!"

clean-notebooks:
	@echo "🧹 Cleaning up notebooks..."
	rm -rf docs/notebooks
	@echo "✅ Notebooks cleaned!"

clean-test-coverage:
	@echo "🧹 Cleaning up test coverage..."
	rm -rf htmlcov .coverage .pytest_cache
	@echo "✅ Test coverage cleaned!"

# ==============================================================================
# PHONY TARGETS
# ==============================================================================

.PHONY: build build-config build-engine build-interface \
        check-all check-all-fix check-config check-engine check-interface \
        check-license-headers \
        clean clean-dist clean-notebooks clean-pycache clean-test-coverage \
        convert-execute-notebooks coverage coverage-config coverage-engine coverage-interface \
        format format-check format-check-config format-check-engine format-check-interface \
        format-config format-engine format-interface \
        generate-colab-notebooks help \
        install install-dev install-dev-notebooks \
        lint lint-config lint-engine lint-fix lint-fix-config lint-fix-engine lint-fix-interface lint-interface \
        perf-import publish serve-docs-locally show-versions \
        test test-config test-config-isolated test-e2e test-engine test-engine-isolated \
        test-interface test-interface-isolated test-isolated \
        test-run-all-examples test-run-recipes test-run-tutorials \
        update-license-headers verify-imports
