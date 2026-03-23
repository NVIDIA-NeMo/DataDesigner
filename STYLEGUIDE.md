# Style Guide

This document is the authoritative reference for code style, naming, type annotations, import patterns, and design principles in DataDesigner. It is extracted from the project's coding standards and enforced by `ruff` (>=0.14.10).

For architectural invariants and project identity, see [AGENTS.md](AGENTS.md).
For development workflow and testing, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## General Formatting

- **Line length**: Maximum 120 characters per line
- **Quote style**: Always use double quotes (`"`) for strings
- **Indentation**: Use 4 spaces (never tabs)
- **Target version**: Python 3.10+

## License Headers

All Python files must include the NVIDIA SPDX license header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Use `make update-license-headers` to add headers to all files automatically.

## Future Annotations

Include `from __future__ import annotations` at the top of every Python source file (after the license header) for deferred type evaluation.

## Comments

Only insert comments when code is especially important to understand. For basic code blocks, comments aren't necessary. We want readable code without vacuous comments.

---

## Type Annotations

Type annotations are REQUIRED for all code in this project. This is strictly enforced for code quality and maintainability. Modern type syntax is enforced by ruff rules `UP006`, `UP007`, and `UP045`.

- **ALWAYS** add type annotations to all functions, methods, and class attributes (including tests)
- Use primitive types when possible: `list` not `List`, `dict` not `Dict`, `set` not `Set`, `tuple` not `Tuple` (enforced by `UP006`)
- Use modern union syntax with `|` for optional and union types:
  - `str | None` not `Optional[str]` (enforced by `UP045`)
  - `int | str` not `Union[int, str]` (enforced by `UP007`)
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

---

## Import Style

- **ALWAYS** use absolute imports, never relative imports (enforced by `TID`)
- Place imports at module level, not inside functions (exception: unavoidable for performance reasons)
- Import sorting is handled by `ruff`'s `isort` — imports should be grouped and sorted:
  1. Standard library imports
  2. Third-party imports (use `lazy_heavy_imports` for heavy libraries)
  3. First-party imports (`data_designer`)
- Use standard import conventions (enforced by `ICN`)

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

**Heavy third-party libraries** (>100ms import cost) should be lazy-loaded via `lazy_heavy_imports.py`:

```python
# Don't import directly
import pandas as pd
import numpy as np

# Use lazy loading with IDE support
from typing import TYPE_CHECKING
from data_designer.lazy_heavy_imports import pd, np

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
```

This pattern provides:
- Runtime lazy loading (fast startup)
- Full IDE support (autocomplete, type hints)
- Type checker validation

See [lazy_heavy_imports.py](packages/data-designer-config/src/data_designer/lazy_heavy_imports.py) for the current list of lazy-loaded libraries.

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
       import your_library_name as your_lib
   ```

3. **Verify with performance test:**
   ```bash
   make perf-import CLEAN=1
   ```

#### TYPE_CHECKING Rules

`TYPE_CHECKING` blocks defer imports that are only needed for type hints, preventing circular dependencies and reducing import time.

**DO put in TYPE_CHECKING:**
- Internal `data_designer` imports used **only** in type hints
- Imports that would cause circular dependencies
- Full imports of lazy-loaded libraries for IDE support (e.g., `import pandas as pd` in addition to runtime `from data_designer.lazy_heavy_imports import pd`)

**DON'T put in TYPE_CHECKING:**
- Standard library imports (`Path`, `Any`, `Callable`, `Literal`, `TypeAlias`, etc.)
- Pydantic model types used in field definitions (needed at runtime for validation)
- Types used in discriminated unions (Pydantic needs them at runtime)
- Any import used at runtime (instantiation, method calls, base classes, etc.)

**Examples:**

```python
# CORRECT - Lazy-loaded library with IDE support
from typing import TYPE_CHECKING
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# CORRECT - Standard library NOT in TYPE_CHECKING
from pathlib import Path
from typing import Any

def process_file(path: Path) -> Any:
    return path.read_text()

# CORRECT - Internal type-only import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer.engine.models.facade import ModelFacade

def get_model(model: ModelFacade) -> str:
    return model.name

# INCORRECT - Pydantic field type in TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer.config.models import ModelConfig  # Wrong!

class MyConfig(BaseModel):
    model: ModelConfig  # Pydantic needs this at runtime!

# CORRECT - Pydantic field type at runtime
from data_designer.config.models import ModelConfig

class MyConfig(BaseModel):
    model: ModelConfig
```

---

## Naming Conventions (PEP 8)

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes**: prefix with single underscore `_private_var`
- **Function and method names must start with an action verb**: e.g. `get_value_from` not `value_from`, `coerce_to_int` not `to_int`, `extract_usage` not `usage`

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

---

## Code Organization

- **Public before private**: Public functions/methods appear before private ones in modules and classes
- **Class method order**: `__init__` and other dunder methods first, then properties, then public methods, then private helpers. Group related method types together (e.g., all `@staticmethod`s in one block, all `@classmethod`s in one block).
- **Prefer public over private for testability**: Use public functions (no `_` prefix) for helpers that benefit from direct testing
- **Section comments in larger modules**: Use `# ---` separators to delineate logical groups (e.g. image parsing, usage extraction, generic accessors)

---

## Design Principles

**DRY**
- Extract shared logic into pure helper functions rather than duplicating across similar call sites
- Rule of thumb: tolerate duplication until the third occurrence, then extract

**KISS**
- Prefer flat, obvious code over clever abstractions — two similar lines is better than a premature helper
- When in doubt between DRY and KISS, favor readability over deduplication

**YAGNI**
- Don't add parameters, config, or abstraction layers for hypothetical future use cases
- Don't generalize until the third caller appears

**SOLID**
- Wrap third-party exceptions at module boundaries — callers depend on canonical error types, not leaked internals
- Use `Protocol` for contracts between layers
- One function, one job — separate logic from I/O

---

## Common Pitfalls to Avoid

1. **Mutable default arguments**:

   ```python
   # Bad
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
   # Bad
   from pathlib import Path
   from typing import Any  # Not used

   def process() -> None:
       pass

   # Good
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
   ```

4. **Use comprehensions properly**:

   ```python
   # Bad
   list([x for x in items])  # Unnecessary list() call

   # Good
   [x for x in items]
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

---

## Active Linter Rules

The following ruff linter rules are currently enabled (see [pyproject.toml](pyproject.toml)):

- `W`: pycodestyle warnings
- `F`: pyflakes (unused imports, undefined names)
- `I`: isort (import sorting)
- `ICN`: flake8-import-conventions (standard import names)
- `PIE`: flake8-pie (miscellaneous lints)
- `TID`: flake8-tidy-imports (bans relative imports)
- `UP006`: `List[A]` -> `list[A]`
- `UP007`: `Union[A, B]` -> `A | B`
- `UP045`: `Optional[A]` -> `A | None`

**Note**: Additional rules (E, N, ANN, B, C4, DTZ, RET, SIM, PTH) are commented out but may be enabled in the future. Write code that would pass these checks for future-proofing.
