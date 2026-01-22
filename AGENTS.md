# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

**DataDesigner** is an NVIDIA NeMo project for creating synthetic datasets from scratch. It's a comprehensive framework that generates structured data using multiple generation strategies:

- **Sampled data**: Built-in generators (UUID, DateTime, etc.) and Faker integration
- **LLM-generated content**: Text, code, and structured data via LiteLLM
- **Expression-based columns**: Derived columns using Jinja2 templates
- **Validation & scoring**: Python, SQL, and remote validators; LLM-based judge scoring
- **Seed dataset-based generation**: Generate from existing datasets

### Architecture

The project follows a layered architecture:

1. **Config Layer** ([src/data_designer/config/](src/data_designer/config/)): User-facing configuration API
   - `config_builder.py`: Main builder API for constructing configurations
   - `columns.py`: Column configuration types (Sampler, LLMText, LLMCode, LLMStructured, LLMJudge, Expression, Validation, SeedDataset)
   - `models.py`: Model configurations and inference parameters
   - `sampler_params.py`: Parametrized samplers (Uniform, Category, Person, DateTime, etc.)

2. **Engine Layer** ([src/data_designer/engine/](src/data_designer/engine/)): Internal generation and processing
   - `column_generators/`: Generates individual columns from configs
   - `dataset_builders/`: Orchestrates full dataset generation with DAG-based dependency management
   - `models/`: LLM integration via LiteLLM with response parsing
   - `validators/`: Column validation (Python, SQL, Code, Remote)
   - `sampling_gen/`: Sophisticated person/entity sampling

3. **Interface Layer** ([src/data_designer/interface/](src/data_designer/interface/)): Public API
   - `data_designer.py`: Main `DataDesigner` class (primary entry point)
   - `results.py`: Result containers
   - `errors.py`: Public error types

4. **Essentials** ([src/data_designer/essentials/](src/data_designer/essentials/)): Convenience module re-exporting key classes for users

### Key Design Patterns

- **Builder pattern**: Configuration construction via `DataDesignerConfigBuilder`
- **Registry pattern**: Plugin system for column generators, validators, and profilers
- **Strategy pattern**: Multiple generation approaches (sampled, LLM, expression, seed)
- **DAG-based execution**: Column dependencies managed as directed acyclic graph

## Development Workflow

This project uses `uv` for dependency management and `make` for common tasks:

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras

# Run the main module (if applicable)
uv run python -m data_designer
```

### Code Quality

```bash
# Using Make (recommended)
make lint           # Run ruff linter
make lint-fix       # Fix linting issues automatically
make format         # Format code with ruff
make format-check   # Check code formatting without changes
make check-all      # Run all checks (format-check + lint)
make check-all-fix  # Run all checks with autofix (format + lint-fix)

# Direct commands
uv run ruff check                # Lint all files
uv run ruff check --fix          # Lint with autofix
uv run ruff format               # Format all files
uv run ruff format --check       # Check formatting
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/config/test_sampler_constraints.py

# Run tests with coverage
uv run pytest --cov=data_designer --cov-report=term-missing --cov-report=html

# Using Make
make test           # Run all tests
make coverage       # Run tests with coverage report
```

## Key Files

- [src/data_designer/interface/data_designer.py](src/data_designer/interface/data_designer.py) - Main entry point (`DataDesigner` class)
- [src/data_designer/config/config_builder.py](src/data_designer/config/config_builder.py) - Configuration API (`DataDesignerConfigBuilder`)
- [src/data_designer/engine/dataset_builders/column_wise_builder.py](src/data_designer/engine/dataset_builders/column_wise_builder.py) - Generation orchestrator
- [src/data_designer/essentials/\_\_init\_\_.py](src/data_designer/essentials/__init__.py) - User-facing API exports
- [pyproject.toml](pyproject.toml) - Project dependencies and tool configurations
- [Makefile](Makefile) - Common development commands

## Working Guidelines

- **Comments**: Only insert comments when code is especially important to understand. For basic code blocks, comments aren't necessary. We want readable code without vacuous comments.
- **License headers**: All Python files must include the NVIDIA SPDX license header:
  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```
  Use `make update-license-headers` to add headers to all files automatically.
- **Imports**: Avoid importing Python modules inside method definitions. Prefer module-level imports for better performance and clarity.
- **Type annotations**: ALWAYS add type annotations to all functions, methods, and class attributes (including tests).

## Code Style

This project uses `ruff` (v0.12.3) for linting and formatting. Follow these guidelines to avoid linter errors:

### General Formatting

- **Line length**: Maximum 120 characters per line
- **Quote style**: Always use double quotes (`"`) for strings
- **Indentation**: Use 4 spaces (never tabs)
- **Target version**: Python 3.11+

### Type Annotations

Type annotations are REQUIRED for all code in this project. This is strictly enforced for code quality and maintainability.

- **ALWAYS** add type annotations to all functions, methods, and class attributes (including tests)
- Use primitive types when possible: `list` not `List`, `dict` not `Dict`, `set` not `Set`, `tuple` not `Tuple`
- Use modern union syntax with `|` for optional and union types (Python 3.10+):
  - `str | None` not `Optional[str]`
  - `int | str` not `Union[int, str]`
- Only import from `typing` when absolutely necessary for complex generic types
- For Pydantic models, use field-level type annotations

  ```python
  # Good
  def process_items(items: list[str], max_count: int | None = None) -> dict[str, int]:
      return {item: len(item) for item in items}

  # Avoid - missing type annotations
  def process_items(items, max_count=None):
      return {item: len(item) for item in items}

  # Avoid - old-style typing
  from typing import List, Dict, Optional
  def process_items(items: List[str], max_count: Optional[int] = None) -> Dict[str, int]:
      return {item: len(item) for item in items}
  ```

### Import Style

- **ALWAYS** use absolute imports, never relative imports
- Place imports at module level, not inside functions (exception: it is unavoidable for performance reasons)
- Import sorting is handled by `ruff`'s `isort` - imports should be grouped and sorted:
  1. Standard library imports
  2. Third-party imports (use `lazy_heavy_imports` for heavy libraries)
  3. First-party imports (`data_designer`)
- Use standard import conventions (enforced by `ICN`)
- See [Lazy Loading and TYPE_CHECKING](#lazy-loading-and-type_checking) section for optimization guidelines

  ```python
  # Good
  from data_designer.config.config_builder import DataDesignerConfigBuilder

  # Bad - relative import (will cause linter errors)
  from .config_builder import DataDesignerConfigBuilder

  # Good - imports at module level
  from pathlib import Path

  def process_file(filename: str) -> None:
      path = Path(filename)

  # Bad - import inside function
  def process_file(filename: str) -> None:
      from pathlib import Path
      path = Path(filename)
  ```

### Lazy Loading and TYPE_CHECKING

This project uses lazy loading for heavy third-party dependencies to optimize import performance.

#### When to Use Lazy Loading

**Heavy third-party libraries** (>100ms import cost) should be lazy-loaded via `lazy_heavy_imports.py`:

```python
# ❌ Don't import directly
import pandas as pd
import numpy as np

# ✅ Use lazy loading with IDE support
from typing import TYPE_CHECKING
from data_designer.lazy_heavy_imports import pd, np

if TYPE_CHECKING:
    import pandas as pd  # For IDE autocomplete and type hints
    import numpy as np
```

This pattern provides:
- Runtime lazy loading (fast startup)
- Full IDE support (autocomplete, type hints)
- Type checker validation

**See [lazy_heavy_imports.py](src/data_designer/lazy_heavy_imports.py) for the current list of lazy-loaded libraries.**

#### Adding New Heavy Dependencies

If you add a new dependency with significant import cost (>100ms):

1. **Add to `lazy_heavy_imports.py`:**
   ```python
   _LAZY_IMPORTS = {
       # ... existing entries ...
       "your_lib": "your_library_name",
   }
   ```

2. **Update imports across codebase:**
   ```python
   from typing import TYPE_CHECKING
   from data_designer.lazy_heavy_imports import your_lib

   if TYPE_CHECKING:
       import your_library_name as your_lib  # For IDE support
   ```

3. **Verify with performance test:**
   ```bash
   make perf-import CLEAN=1
   ```

#### Using TYPE_CHECKING Blocks

`TYPE_CHECKING` blocks defer imports that are only needed for type hints, preventing circular dependencies and reducing import time.

**For internal data_designer imports:**

```python
from __future__ import annotations  # Always include at top

from typing import TYPE_CHECKING

# Runtime imports
from pathlib import Path
from data_designer.config.base import ConfigBase

if TYPE_CHECKING:
    # Type-only imports - only visible to type checkers
    from data_designer.engine.models.facade import ModelFacade

def get_model(model: ModelFacade) -> str:
    return model.name
```

**For lazy-loaded libraries (see pattern in "When to Use Lazy Loading" above):**
- Import from `lazy_heavy_imports` for runtime
- Add full import in `TYPE_CHECKING` block for IDE support

**Rules for TYPE_CHECKING:**

✅ **DO put in TYPE_CHECKING:**
- Internal `data_designer` imports used **only** in type hints
- Imports that would cause circular dependencies
- **Full imports of lazy-loaded libraries for IDE support** (e.g., `import pandas as pd` in addition to runtime `from data_designer.lazy_heavy_imports import pd`)

❌ **DON'T put in TYPE_CHECKING:**
- **Standard library imports** (`Path`, `Any`, `Callable`, `Literal`, `TypeAlias`, etc.)
- **Pydantic model types** used in field definitions (needed at runtime for validation)
- **Types used in discriminated unions** (Pydantic needs them at runtime)
- **Any import used at runtime** (instantiation, method calls, base classes, etc.)

**Examples:**

```python
# ✅ CORRECT - Lazy-loaded library with IDE support
from typing import TYPE_CHECKING
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd  # IDE gets full type hints

def load_data(path: str) -> pd.DataFrame:  # IDE understands pd.DataFrame
    return pd.read_csv(path)

# ✅ CORRECT - Standard library NOT in TYPE_CHECKING
from pathlib import Path
from typing import Any

def process_file(path: Path) -> Any:
    return path.read_text()

# ✅ CORRECT - Internal type-only import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer.engine.models.facade import ModelFacade

def get_model(model: ModelFacade) -> str:  # Only used in type hint
    return model.name

# ❌ INCORRECT - Pydantic field type in TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer.config.models import ModelConfig  # Wrong!

class MyConfig(BaseModel):
    model: ModelConfig  # Pydantic needs this at runtime!

# ✅ CORRECT - Pydantic field type at runtime
from data_designer.config.models import ModelConfig

class MyConfig(BaseModel):
    model: ModelConfig
```

### Naming Conventions (PEP 8)

Follow PEP 8 naming conventions:

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes**: prefix with single underscore `_private_var`

  ```python
  # Good
  class DatasetGenerator:
      MAX_RETRIES = 3

      def __init__(self) -> None:
          self._cache: dict[str, str] = {}

      def generate_dataset(self, config: dict[str, str]) -> pd.DataFrame:
          pass

  # Bad
  class dataset_generator:  # Should be PascalCase
      maxRetries = 3        # Should be UPPER_SNAKE_CASE

      def GenerateDataset(self, Config):  # Should be snake_case
          pass
  ```

### Common Pitfalls to Avoid

1. **Mutable default arguments**:

   ```python
   # Bad - mutable default argument
   def add_item(item: str, items: list[str] = []) -> list[str]:
       items.append(item)
       return items

   # Good
   def add_item(item: str, items: list[str] | None = None) -> list[str]:
       if items is None:
           items = []
       items.append(item)
       return items
   ```

2. **Unused imports and variables**:

   ```python
   # Bad - unused import
   from pathlib import Path
   from typing import Any  # Not used

   def process() -> None:
       pass

   # Good - only import what you use
   from pathlib import Path

   def process() -> None:
       pass
   ```

3. **Simplify code where possible** (enforced by `SIM`):

   ```python
   # Bad
   if condition:
       return True
   else:
       return False

   # Good
   return condition

   # Bad
   if key in my_dict:
       value = my_dict[key]
   else:
       value = default

   # Good
   value = my_dict.get(key, default)
   ```

4. **Use comprehensions properly**:

   ```python
   # Bad
   list([x for x in items])  # Unnecessary list() call

   # Good
   [x for x in items]

   # Bad
   dict([(k, v) for k, v in items])

   # Good
   {k: v for k, v in items}
   ```

5. **Proper return statements**:

   ```python
   # Bad - unnecessary else after return
   def get_value(condition: bool) -> str:
       if condition:
           return "yes"
       else:
           return "no"

   # Good
   def get_value(condition: bool) -> str:
       if condition:
           return "yes"
       return "no"
   ```

### Active Linter Rules

The following ruff linter rules are currently enabled (see [pyproject.toml](pyproject.toml)):

- `W`: pycodestyle warnings
- `F`: pyflakes (unused imports, undefined names)
- `I`: isort (import sorting)
- `ICN`: flake8-import-conventions (standard import names)
- `PIE`: flake8-pie (miscellaneous lints)

**Note**: Additional rules (E, N, UP, ANN, B, C4, DTZ, RET, SIM, PTH) are commented out but may be enabled in the future. Write code that would pass these checks for future-proofing.

## Testing Patterns

The project uses `pytest` with the following patterns:

- **Fixtures**: Shared test data and configurations in [tests/conftest.py](tests/conftest.py)
- **Stub configs**: YAML-based configuration stubs for testing (see `stub_data_designer_config_str` fixture)
- **Mocking**: Use `unittest.mock.patch` for external services and dependencies
- **Async support**: pytest-asyncio for async tests (`asyncio_default_fixture_loop_scope = "session"`)
- **HTTP mocking**: pytest-httpx for mocking HTTP requests
- **Coverage**: Track test coverage with pytest-cov

Example test structure:

```python
import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder

def test_something(stub_model_configs):
    """Test description."""
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    # ... test implementation
    assert expected == actual
```

## Column Configuration Types

When working with column configurations, understand these key types:

- **`SamplerColumnConfig`**: Built-in samplers (UUID, Category, Uniform, Gaussian, Person, DateTime, etc.)
- **`LLMTextColumnConfig`**: LLM text generation with Jinja2 templating
- **`LLMCodeColumnConfig`**: Code generation with language specification
- **`LLMStructuredColumnConfig`**: Structured JSON generation with schema
- **`LLMJudgeColumnConfig`**: Judge/scoring columns for quality assessment
- **`ExpressionColumnConfig`**: Expression-based derived columns (Python eval or Jinja2)
- **`ValidationColumnConfig`**: Validation results (Python, SQL, Code, Remote validators)
- **`SeedDatasetColumnConfig`**: Data from seed datasets

See [src/data_designer/config/columns.py](src/data_designer/config/columns.py) for detailed schemas.

## Model Configuration

Models are configured via `ModelConfig` with:

- `alias`: User-defined alias for the model
- `model`: Model ID (e.g., from build.nvidia.com)
- `inference_parameters`: Temperature, top_p, max_tokens (can be distribution-based)
- `system_prompt`: Optional system prompt
- `image_modality`: Support for image inputs

See [src/data_designer/config/models.py](src/data_designer/config/models.py) for details.

## Registry System

The project uses a registry pattern for extensibility. Key registries:

- **Column generators**: [src/data_designer/engine/column_generators/registry.py](src/data_designer/engine/column_generators/registry.py)
- **Validators**: [src/data_designer/engine/validators/](src/data_designer/engine/validators/)
- **Column profilers**: [src/data_designer/engine/analysis/column_profilers/registry.py](src/data_designer/engine/analysis/column_profilers/registry.py)
- **Models**: [src/data_designer/engine/models/registry.py](src/data_designer/engine/models/registry.py)

When adding new generators or validators, register them appropriately.

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

## Common Development Tasks

```bash
# Clean up generated files
make clean

# Update license headers
make update-license-headers

# Run all checks before committing
make check-all-fix
make test

# Generate coverage report
make coverage
# View htmlcov/index.html in browser
```

## Additional Resources

- **README.md**: Installation and basic usage examples
- **src/data_designer/config/**: Configuration API documentation
- **tests/**: Comprehensive test suite with usage examples
