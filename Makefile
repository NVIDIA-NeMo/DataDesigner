REPO_PATH := $(shell pwd)

help:
	@echo "Available commands:"
	@echo "  clean                  - Remove coverage reports and cache files"
	@echo "  coverage               - Run tests with coverage report"
	@echo "  test                   - Run all unit tests"
	@echo "  update-license-headers - Add license headers to all files"

clean:
	@echo "🧹 Cleaning up coverage reports and cache files..."
	rm -rf htmlcov .coverage .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

coverage:
	@echo "📊 Running tests with coverage analysis..."
	uv run pytest --cov=data_designer --cov-report=term-missing --cov-report=html
	@echo "✅ Coverage report generated in htmlcov/index.html"

test:
	@echo "🧪 Running unit tests..."
	uv run pytest

update-license-headers:
	@echo "🔍 Updating license headers in all files..."
	uv run python $(REPO_PATH)/scripts/add-license-headers.py