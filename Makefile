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
	@echo "📦 Installation:"
	@echo "  install                - Install project dependencies with uv"
	@echo "  install-dev            - Install project with dev dependencies"
	@echo "  install-dev-notebooks  - Install dev + notebook dependencies (Jupyter, etc.)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test                   - Run all unit tests"
	@echo "  coverage               - Run tests with coverage report"
	@echo ""
	@echo "✨ Code Quality:"
	@echo "  format                 - Format code with ruff"
	@echo "  format-check           - Check code formatting without making changes"
	@echo "  lint                   - Lint code with ruff"
	@echo "  lint-fix               - Fix linting issues automatically"
	@echo ""
	@echo "🔍 Combined Checks:"
	@echo "  check-all              - Run all checks (format-check + lint)"
	@echo "  check-all-fix          - Run all checks with autofix (format + lint-fix)"
	@echo ""
	@echo "🛠️  Utilities:"
	@echo "  clean                  - Remove coverage reports and cache files"
	@echo "  serve-docs-locally     - Serve documentation locally"
	@echo "  check-license-headers  - Check if all files have license headers"
	@echo "  update-license-headers - Add license headers to all files"
	@echo ""
	@echo "═════════════════════════════════════════════════════════════"
	@echo "💡 Tip: Run 'make <command>' to execute any command above"
	@echo ""

clean:
	@echo "🧹 Cleaning up coverage reports and cache files..."
	rm -rf htmlcov .coverage .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

coverage:
	@echo "📊 Running tests with coverage analysis..."
	uv run --group dev pytest --cov=data_designer --cov-report=term-missing --cov-report=html
	@echo "✅ Coverage report generated in htmlcov/index.html"

check-all: format-check lint
	@echo "✅ All checks complete!"

check-all-fix: format lint-fix
	@echo "✅ All checks with autofix complete!"

format:
	@echo "📐 Formatting code with ruff..."
	uv run ruff format src/ tests/ scripts/ --exclude '**/src/data_designer/_version.py'
	@echo "✅ Formatting complete!"

format-check:
	@echo "📐 Checking code formatting with ruff..."
	uv run ruff format --check src/ tests/ scripts/ --exclude '**/src/data_designer/_version.py'
	@echo "✅ Formatting check complete! Run 'make format' to auto-fix issues."

lint:
	@echo "🔍 Linting code with ruff..."
	uv run ruff check --output-format=full src/ tests/ scripts/ --exclude '**/src/data_designer/_version.py'
	@echo "✅ Linting complete! Run 'make lint-fix' to auto-fix issues."

lint-fix:
	@echo "🔍 Fixing linting issues with ruff..."
	uv run ruff check --fix src/ tests/ scripts/ --exclude '**/src/data_designer/_version.py'
	@echo "✅ Linting with autofix complete!"

test:
	@echo "🧪 Running unit tests..."
	uv run --group dev pytest

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

install:
	@echo "📦 Installing project dependencies..."
	uv sync
	@echo "✅ Installation complete!"

install-dev:
	@echo "📦 Installing project with dev dependencies..."
	uv sync --group dev
	$(call install-pre-commit-hooks)
	@echo "✅ Dev installation complete!"

install-dev-notebooks:
	@echo "📦 Installing project with notebook dependencies..."
	uv sync --group dev --group notebooks
	$(call install-pre-commit-hooks)
	@echo "✅ Dev + notebooks installation complete!"

.PHONY: clean coverage format format-check lint lint-fix test check-license-headers update-license-headers check-all check-all-fix install install-dev install-dev-notebooks
