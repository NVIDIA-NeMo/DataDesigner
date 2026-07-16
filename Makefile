# ==============================================================================
# VARIABLES AND FUNCTIONS
# ==============================================================================

REPO_PATH := $(shell pwd)
PRE_COMMIT ?= .venv/bin/pre-commit
LICENSE_PYTHON_VERSION ?= 3.11

# Package directories
CONFIG_PKG := packages/data-designer-config
ENGINE_PKG := packages/data-designer-engine
INTERFACE_PKG := packages/data-designer

# Package source and test paths
CONFIG_PATHS := $(CONFIG_PKG)/src $(CONFIG_PKG)/tests
ENGINE_PATHS := $(ENGINE_PKG)/src $(ENGINE_PKG)/tests
INTERFACE_PATHS := $(INTERFACE_PKG)/src $(INTERFACE_PKG)/tests $(INTERFACE_PKG)/dev-tools
ALL_PKG_PATHS := packages/ scripts/ tests_e2e/

# Test directories
CONFIG_TESTS := $(CONFIG_PKG)/tests
ENGINE_TESTS := $(ENGINE_PKG)/tests
INTERFACE_TESTS := $(INTERFACE_PKG)/tests

define install-pre-commit-hooks
	@if [ ! -f $(REPO_PATH)/.git/hooks/pre-commit ]; then \
		echo "🪝 Installing pre-commit hooks..."; \
		$(PRE_COMMIT) install; \
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
	@echo "  install-dev-notebooks     - Install all packages + dev + docs + notebook tools"
	@echo "  install-dev-recipes       - Install all packages + dev + recipe dependencies"
	@echo ""
	@echo "🧪 Testing (all packages):"
	@echo "  test                      - Run all unit tests"
	@echo "  coverage                  - Run tests with coverage report"
	@echo "  test-e2e                  - Run e2e plugin tests"
	@echo "  health-checks             - Run provider health checks"
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
	@echo "  convert-execute-notebooks - Convert notebooks from .py to .ipynb using jupytext (USE_CACHE=1 to skip unchanged)"
	@echo "  generate-colab-notebooks  - Generate Colab-compatible notebooks"
	@echo "  generate-fern-notebooks   - Convert docs/notebook_source/*.py → fern/components/notebooks/{json,ts}"
	@echo "  generate-fern-notebooks-with-outputs - Full pipeline: execute notebooks (needs API key), colabify, convert to Fern"
	@echo "  install-docs-deps        - Install docs and notebook dependencies"
	@echo "  prepare-fern-release VERSION=X.Y.Z - Add or refresh Fern version files for release preview"
	@echo "  check-fern-release-version VERSION=X.Y.Z - Verify Fern has a version entry for release publishing"
	@echo "  prepare-fern-docs         - Generate local Fern artifacts"
	@echo "  check-fern-docs           - Generate local Fern artifacts and run fern check"
	@echo "  check-fern-docs-locally   - Install deps, generate Fern artifacts, and run fern check"
	@echo "  serve-fern-docs-locally   - Generate local Fern artifacts and serve Fern docs"
	@echo "  check-license-headers     - Check if all files have license headers"
	@echo "  check-dependency-licenses - Check runtime dependency license compatibility"
	@echo "  update-license-headers    - Add license headers to all files"
	@echo ""
	@echo "⚡ Performance:"
	@echo "  perf-import               - Profile pure import time and show summary"
	@echo "  perf-import CLEAN=1       - Clean cache, then profile pure import time"
	@echo "  perf-import NOFILE=1      - Profile pure import without writing to file (for CI)"
	@echo "  perf-import-runtime       - Profile runtime init time (constructors included)"
	@echo "  bench-cli-startup         - Benchmark CLI startup (isolated venv)"
	@echo "  bench-cli-startup-verbose - Benchmark CLI startup with import trace"
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
	@echo "   Groups: dev + docs + notebooks (Jupyter, jupytext, etc.)"
	uv sync --all-packages --group dev --group docs --group notebooks
	$(call install-pre-commit-hooks)
	@echo "✅ Dev + docs + notebooks installation complete!"
	@echo ""
	@echo "💡 Run 'make test-run-tutorials' to test notebook tutorials"

install-dev-recipes:
	@echo "📦 Installing DataDesigner workspace with recipe dependencies..."
	@echo "   Packages: data-designer-config → data-designer-engine → data-designer"
	@echo "   Groups: dev + recipes (bm25s, pymupdf, etc.)"
	uv sync --all-packages --group dev --group recipes
	$(call install-pre-commit-hooks)
	@echo "✅ Dev + recipes installation complete!"
	@echo ""
	@echo "💡 Run 'make test-run-recipes' to test recipe scripts"

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
	@echo "🧹 Cleaning e2e test environment..."
	rm -rf tests_e2e/uv.lock tests_e2e/__pycache__ tests_e2e/.venv
	@echo "🧪 Running e2e tests..."
	uv run --no-cache --refresh --directory tests_e2e pytest -s

health-checks:
	@echo "🏥 Running provider health checks..."
	uv run --group dev python scripts/health_checks.py

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
		(cd "$$RECIPE_WORKDIR" && uv run --project "$(REPO_PATH)" --group notebooks --group recipes python "$(REPO_PATH)/$$f" --model-alias nvidia-text --artifact-path "$$RECIPE_WORKDIR" --num-records 5) || exit 1; \
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
# LICENSE CHECKS
# ==============================================================================

check-dependency-licenses:
	@echo "🔍 Checking Python $(LICENSE_PYTHON_VERSION) locked runtime dependency licenses..."
	uv run --isolated --python $(LICENSE_PYTHON_VERSION) --all-packages --no-dev --group license-check --locked \
		python $(REPO_PATH)/scripts/check_dependency_licenses.py

check-license-headers:
	@echo "🔍 Checking license headers in all files..."
	uv run python $(REPO_PATH)/scripts/update_license_headers.py --check

update-license-headers:
	@echo "🔍 Updating license headers in all files..."
	uv run python $(REPO_PATH)/scripts/update_license_headers.py

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

# Pin docs setup to a Python version covered by all docs/notebook deps. We
# default to 3.13 to match the published docs builds; override with
# DOCS_PYTHON_VERSION=... to test on another interpreter.
DOCS_PYTHON_VERSION ?= 3.13
DOCS_PYTHON ?= .venv/bin/python
DOCS_JUPYTEXT ?= .venv/bin/jupytext
FERN_VERSION ?= $(shell jq -r .version fern/fern.config.json)
FERN ?= npx -y fern-api@$(FERN_VERSION)
FERN_ORG ?= $(shell jq -r .organization fern/fern.config.json)
FERN_GLOBAL_THEME ?= $(shell awk '/^global-theme:/ { print $$2; exit }' fern/docs.yml)

# Route urllib/requests/httpx through certifi's CA bundle. Necessary when uv
# resolves $(DOCS_PYTHON_VERSION) to a python.org installer build, which ships without
# populated CA certs (notebook 3 downloads a CSV over HTTPS at exec time).
DOCS_CERTS = SSL_CERT_FILE=$$($(DOCS_PYTHON) -c "import certifi; print(certifi.where())") \
             REQUESTS_CA_BUNDLE=$$($(DOCS_PYTHON) -c "import certifi; print(certifi.where())")

install-docs-deps:
	@echo "📦 Installing docs dependencies (Python $(DOCS_PYTHON_VERSION))..."
	uv sync --python $(DOCS_PYTHON_VERSION) --all-packages --group docs --group notebooks

prepare-fern-release:
ifndef VERSION
	$(error VERSION is required, e.g. make prepare-fern-release VERSION=0.5.10)
endif
	$(DOCS_PYTHON) fern/scripts/fern-release-version.py prepare --version $(VERSION) $(if $(FORCE),--force,)

check-fern-release-version:
ifndef VERSION
	$(error VERSION is required, e.g. make check-fern-release-version VERSION=0.5.10)
endif
	$(DOCS_PYTHON) fern/scripts/fern-release-version.py check --version $(VERSION) $(if $(REQUIRE_LATEST),--require-latest-matches-release,)

prepare-fern-docs: generate-fern-notebooks
	@echo "✅ Fern local artifacts ready"

check-fern-docs: prepare-fern-docs
	cd fern && $(FERN) check

check-fern-docs-locally:
	@$(MAKE) install-docs-deps
	@$(MAKE) check-fern-docs
	@echo "✅ Fern docs check complete"

check-fern-theme-access:
	@if [ -n "$(FERN_GLOBAL_THEME)" ]; then \
		cd fern && $(FERN) docs theme list --org $(FERN_ORG) --json >/dev/null || { \
			echo ""; \
			echo "Unable to access Fern global theme '$(FERN_GLOBAL_THEME)' for org '$(FERN_ORG)'."; \
			echo "Fix: sign in to https://dashboard.buildwithfern.com, then run:"; \
			echo "  cd fern && $(FERN) login"; \
			echo "If it still fails, export a privileged DOCS_FERN_TOKEN as FERN_TOKEN."; \
			exit 1; \
		}; \
	fi

serve-fern-docs-locally:
	@$(MAKE) install-docs-deps
	@$(MAKE) prepare-fern-docs
	@$(MAKE) serve-fern-docs-dev

serve-fern-docs-dev:
	@if $(MAKE) check-fern-theme-access >/dev/null 2>&1; then \
		cd fern && PNPM_CONFIG_DANGEROUSLY_ALLOW_ALL_BUILDS=true $(FERN) docs dev; \
	else \
		echo "Unable to access Fern global theme '$(FERN_GLOBAL_THEME)' for org '$(FERN_ORG)'."; \
		echo "Serving with the local NVIDIA-style fallback. Run 'make check-fern-theme-access' for auth details."; \
		$(MAKE) serve-fern-docs-local-theme; \
	fi

serve-fern-docs-local-theme:
	@PNPM_CONFIG_DANGEROUSLY_ALLOW_ALL_BUILDS=true $(DOCS_PYTHON) fern/scripts/serve-local-docs-preview.py --root fern -- $(FERN) docs dev

convert-execute-notebooks:
ifeq ($(USE_CACHE),1)
	@echo "📓 Converting Python tutorials to notebooks (with caching)..."
	@$(DOCS_CERTS) DOCS_JUPYTEXT=$(DOCS_JUPYTEXT) bash docs/scripts/build_notebooks_cached.sh
else
	@echo "📓 Converting Python tutorials to notebooks and executing ($(DOCS_PYTHON))..."
	@rm -rf docs/notebooks
	@mkdir -p docs/notebooks
	cp docs/notebook_source/_README.md docs/notebooks/README.md
	cp docs/notebook_source/_pyproject.toml docs/notebooks/pyproject.toml
	@$(DOCS_CERTS) bash -c '\
		failed=""; \
		for f in docs/notebook_source/*.py; do \
			[ -f "$$f" ] || continue; \
			echo "▶ executing $$f"; \
			$(DOCS_JUPYTEXT) --to ipynb --execute "$$f" || failed="$$failed\n   • $$f"; \
		done; \
		for f in docs/notebook_source/*.ipynb; do [ -f "$$f" ] && mv "$$f" docs/notebooks/; done; \
		rm -rf docs/notebook_source/artifacts; \
		rm -f docs/notebook_source/*.csv; \
		if [ -n "$$failed" ]; then \
			echo ""; \
			echo "❌ Some notebooks failed (often missing API keys for image/audio providers)."; \
			printf "   Failed:%b\n" "$$failed"; \
			exit 1; \
		fi'
	@echo "✅ Notebooks executed under docs/notebooks/"
endif

generate-colab-notebooks:
	@echo "📓 Generating Colab-compatible notebooks ($(DOCS_PYTHON))..."
	$(DOCS_PYTHON) docs/scripts/generate_colab_notebooks.py
	@echo "✅ Colab notebooks created in docs/colab_notebooks/"

generate-fern-notebooks:
	@echo "📓 Converting notebooks to Fern format for NotebookViewer ($(DOCS_PYTHON))..."
	@mkdir -p fern/components/notebooks
	@failed=; tmp_dir=$$(mktemp -d); trap 'rm -rf "$$tmp_dir"' EXIT; \
	for src in docs/notebook_source/*.py; do \
		[ -f "$$src" ] || continue; \
		name=$$(basename "$$src" .py); \
		if [ -f "docs/notebooks/$$name.ipynb" ]; then \
			input="docs/notebooks/$$name.ipynb"; \
			source_label="notebook"; \
		else \
			input="$$tmp_dir/$$name.ipynb"; \
			source_label="source"; \
			$(DOCS_JUPYTEXT) --to ipynb --output "$$input" "$$src" >/dev/null || failed=1; \
		fi; \
		if [ ! -f "$$input" ]; then \
			echo "❌ Missing notebook input for $$name: $$input"; \
			failed=1; \
			continue; \
		fi; \
		echo "   $$name ($$source_label)"; \
		$(DOCS_PYTHON) fern/scripts/ipynb-to-fern-json.py "$$input" -o fern/components/notebooks/$$name.json || failed=1; \
	done; \
	if [ -n "$$failed" ]; then exit 1; fi
	@echo "✅ Fern notebooks created in fern/components/notebooks/"

generate-fern-notebooks-with-outputs: convert-execute-notebooks generate-colab-notebooks generate-fern-notebooks
	@echo "✅ Full notebook pipeline complete (executed → colab → fern)"

# ==============================================================================
# PERFORMANCE
# ==============================================================================

perf-import:
ifdef CLEAN
	@$(MAKE) clean-pycache
endif
	@echo "⚡ Profiling pure import time for data_designer.config and DataDesigner symbol..."
ifdef NOFILE
	@PERF_OUTPUT=$$(uv run python -X importtime -c "import data_designer.config as dd; from data_designer.interface import DataDesigner" 2>&1); \
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
	uv run python -X importtime -c "import data_designer.config as dd; from data_designer.interface import DataDesigner" > "$$PERF_FILE" 2>&1; \
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

perf-import-runtime:
ifdef CLEAN
	@$(MAKE) clean-pycache
endif
	@echo "⚡ Profiling runtime initialization time (DataDesigner + DataDesignerConfigBuilder constructors)..."
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
	@PERF_FILE="perf_import_runtime_$$(date +%Y%m%d_%H%M%S).txt"; \
	uv run python -X importtime -c "import data_designer.config as dd; from data_designer.interface import DataDesigner; DataDesigner(); dd.DataDesignerConfigBuilder()" > "$$PERF_FILE" 2>&1; \
	echo "📊 Runtime import profile saved to $$PERF_FILE"; \
	echo ""; \
	echo "Summary:"; \
	tail -1 "$$PERF_FILE" | awk '{printf "  Total: %.3fs\n", $$5/1000000}'; \
	echo ""; \
	echo "💡 Top 10 slowest imports:"; \
	printf "%-12s %-12s %s\n" "Self (s)" "Cumulative (s)" "Module"; \
	printf "%-12s %-12s %s\n" "--------" "--------------" "------"; \
	grep "import time:" "$$PERF_FILE" | sort -rn -k5 | head -10 | awk '{printf "%-12.3f %-12.3f %s", $$3/1000000, $$5/1000000, $$7; for(i=8;i<=NF;i++) printf " %s", $$i; printf "\n"}'
endif

BENCH_CLI_ARGS ?=

bench-cli-startup:
	@echo "⚡ Benchmarking CLI startup time (isolated venv)..."
	uv run python scripts/benchmarks/benchmark_cli_startup.py $(BENCH_CLI_ARGS)

bench-cli-startup-verbose:
	@echo "⚡ Benchmarking CLI startup time (isolated + import trace)..."
	uv run python scripts/benchmarks/benchmark_cli_startup.py --verbose $(BENCH_CLI_ARGS)

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

.PHONY: bench-cli-startup bench-cli-startup-verbose \
        build build-config build-engine build-interface \
        check-all check-all-fix check-config check-engine check-interface \
        check-dependency-licenses check-fern-docs check-fern-docs-locally check-fern-release-version check-fern-theme-access check-license-headers \
        clean clean-dist clean-notebooks clean-pycache clean-test-coverage \
        convert-execute-notebooks \
        coverage coverage-config coverage-engine coverage-interface \
        format format-check format-check-config format-check-engine format-check-interface \
        format-config format-engine format-interface \
        generate-colab-notebooks generate-fern-notebooks generate-fern-notebooks-with-outputs help \
        install install-dev install-dev-notebooks install-dev-recipes install-docs-deps \
        lint lint-config lint-engine lint-fix lint-fix-config lint-fix-engine lint-fix-interface lint-interface \
        perf-import perf-import-runtime prepare-fern-docs prepare-fern-release publish serve-fern-docs-dev serve-fern-docs-local-theme serve-fern-docs-locally show-versions \
        health-checks \
        test test-config test-config-isolated test-e2e test-engine test-engine-isolated \
        test-interface test-interface-isolated test-isolated \
        test-run-all-examples test-run-recipes test-run-tutorials \
        update-license-headers verify-imports
