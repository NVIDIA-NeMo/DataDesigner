# Development Guide

This document covers local setup, day-to-day development workflow, testing, and pre-commit usage for DataDesigner contributors.

For architectural invariants and project identity, see [AGENTS.md](AGENTS.md).
For code style, naming, and import conventions, see [STYLEGUIDE.md](STYLEGUIDE.md).
For the contribution workflow (issues, PRs, agent-assisted development), see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for dependency management
- **[GNU Make](https://www.gnu.org/software/make/)** for common tasks

## Local Setup

### Clone and Install

```bash
git clone https://github.com/NVIDIA-NeMo/DataDesigner.git
cd DataDesigner

# Install with dev dependencies
make install-dev

# Or, if you use Jupyter / IPython for development
make install-dev-notebooks
```

### Verify Your Setup

```bash
make test && make check-all
```

If no errors are reported, you're ready to develop.

---

## Day-to-Day Workflow

### Branching

```bash
git checkout main
git pull origin main
git checkout -b <username>/<type>/<issue-number>-<short-description>
```

Branch name types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `style`, `perf`.

Example: `nmulepati/feat/123-add-xyz-generator`

### Syncing with Upstream

If you're working from a fork, add the upstream remote first:

```bash
git remote add upstream https://github.com/NVIDIA-NeMo/DataDesigner.git
```

Then sync:

```bash
git fetch upstream
git merge upstream/main
```

### Validation Before Committing

```bash
make check-all-fix   # format + lint (ruff)
make test            # run all test suites
```

---

## Code Quality

### Using Make (Recommended)

```bash
make lint              # Run ruff linter
make lint-fix          # Fix linting issues automatically
make format            # Format code with ruff
make format-check      # Check code formatting without changes
make check-all         # Run all checks (format-check + lint)
make check-all-fix     # Run all checks with autofix (format + lint-fix)
```

### Direct Commands

```bash
uv run ruff check                # Lint all files
uv run ruff check --fix          # Lint with autofix
uv run ruff format               # Format all files
uv run ruff format --check       # Check formatting
```

---

## Testing

### Running Tests

`make test` runs all three package test suites in sequence (config, engine, interface). When iterating on a single package, run its tests directly:

```bash
# Run all tests (config + engine + interface)
make test

# Run a single package's tests
make test-config       # data-designer-config
make test-engine       # data-designer-engine
make test-interface    # data-designer (interface)

# Run a specific test file
uv run pytest tests/config/test_sampler_constraints.py

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage
make coverage
# View htmlcov/index.html in browser

# E2E and example tests (slower, require API keys — see README.md for setup)
make test-e2e              # end-to-end tests
make test-run-tutorials    # run tutorial notebooks as tests
make test-run-recipes      # run recipe scripts as tests
```

### Test Patterns

The project uses `pytest` with the following patterns:

- **Flat test functions**: Write standalone `test_*` functions, not `class`-based test suites. Use fixtures and parametrize for shared setup instead of class inheritance.
- **Fixtures**: Shared fixtures are provided via `pytest_plugins` from `data_designer.config.testing.fixtures` and `data_designer.engine.testing.fixtures`, plus local `conftest.py` files in each test directory
- **Stub configs**: YAML-based configuration stubs for testing (see `stub_data_designer_config_str` fixture)
- **Mocking**: Use `unittest.mock.patch` for external services and dependencies
- **Async support**: pytest-asyncio for async tests (`asyncio_default_fixture_loop_scope = "session"`)
- **HTTP mocking**: pytest-httpx for mocking HTTP requests
- **Coverage**: Track test coverage with pytest-cov

### Test Guidelines

- **Test public APIs only**: Tests should exercise public interfaces, not `_`-prefixed functions or classes. If something is hard to test without reaching into private internals, consider refactoring the code to expose a public entry point
- **Type annotations required**: Test functions and fixtures must include type annotations — `-> None` for tests, typed parameters, and typed return values for fixtures
- **Imports at module level**: Follow the same import rules as production code — keep imports at the top of the file, not inside test functions
- **Parametrize over duplicate**: Use `@pytest.mark.parametrize` (with `ids=` for readable names) instead of writing multiple test functions for variations of the same behavior
- **Minimal fixtures**: Fixtures should be simple — one fixture, one responsibility, just setup with no behavior logic
- **Shared fixtures in `conftest.py`**: Place fixtures shared across a test directory in `conftest.py`
- **Mock at boundaries**: Mock external dependencies (APIs, databases, third-party services), not internal functions
- **Test behavior, not implementation**: Assert on outputs and side effects, not internal call counts (unless verifying routing)
- **Keep mocking shallow**: If a test requires deeply nested mocking, the code under test may need refactoring

### Example Test

```python
from typing import Any

from data_designer.config.config_builder import DataDesignerConfigBuilder


def test_something(stub_model_configs: dict[str, Any]) -> None:
    """Test description."""
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    # ... test implementation
    assert expected == actual
```

---

## Pre-commit Hooks

The project uses pre-commit hooks to enforce code quality. Install them with:

```bash
uv run pre-commit install
```

Hooks include:
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON/TOML validation
- Merge conflict detection
- Debug statement detection
- Ruff linting and formatting

---

## Common Tasks

```bash
make clean                       # Clean up generated files
make update-license-headers      # Add SPDX headers to new files
make check-all-fix               # Format + lint before committing
make test                        # Run all tests
make coverage                    # Run tests with coverage report
make perf-import                 # Profile import time
make perf-import CLEAN=1         # Clean cache first, then profile
make convert-execute-notebooks   # Regenerate .ipynb from docs/notebook_source/*.py
make generate-colab-notebooks    # Generate Colab-compatible notebooks
```

---

## Import Performance

After adding heavy third-party dependencies, verify import performance:

```bash
make perf-import CLEAN=1
```

There is also a CI test (`test_import_performance` in `packages/data-designer/tests/test_import_perf.py`) that runs 5 import cycles (1 cold + 4 warm) and fails if the average exceeds **3 seconds**. If your dependency causes a regression, add it to `lazy_heavy_imports.py` — see [STYLEGUIDE.md](STYLEGUIDE.md) for the lazy loading pattern.
