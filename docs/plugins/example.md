!!! warning "Experimental Feature"
    The plugin system is currently **experimental** and under active development. The documentation, examples, and plugin interface are subject to significant changes in future releases. If you encounter any issues, have questions, or have ideas for improvement, please consider starting [a discussion on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/discussions).


# Example Plugin: Column Generator

Data Designer supports two plugin types: **column generators** and **seed readers**. This page walks through a complete column generator example.

A Data Designer plugin is implemented as a Python package with three main components:

1. **Configuration Class**: Defines the parameters users can configure
2. **Implementation Class**: Contains the core logic of the plugin
3. **Plugin Object**: Connects the config and implementation classes to make the plugin discoverable

We recommend separating these into individual files (`config.py`, `impl.py`, `plugin.py`) within a plugin subdirectory. This keeps the code organized, makes it easy to test each component independently, and guards against circular dependencies — since the config module can be imported without pulling in the engine-level implementation classes, and the plugin object can be discovered without importing either.

---

## Column Generator Plugin: Index Multiplier

In this section, we will build a simple column generator plugin that generates values by multiplying the row index by a user-specified multiplier.

### Step 1: Create a Python package

We recommend the following structure for column generator plugins:

```
data-designer-index-multiplier/
├── pyproject.toml
└── src/
    └── data_designer_index_multiplier/
        ├── __init__.py
        ├── config.py
        ├── impl.py
        └── plugin.py
```

### Step 2: Create the config class

The configuration class defines what parameters users can set when using your plugin. For column generator plugins, it must inherit from [SingleColumnConfig](../code_reference/column_configs.md#data_designer.config.column_configs.SingleColumnConfig) and include a [discriminator field](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions).

Create `src/data_designer_index_multiplier/config.py`:

```python
from typing import Literal

from data_designer.config.base import SingleColumnConfig


class IndexMultiplierColumnConfig(SingleColumnConfig):
    """Configuration for the index multiplier column generator."""

    # Required: discriminator field with a unique Literal type
    # This value identifies your plugin and becomes its column_type
    column_type: Literal["index-multiplier"] = "index-multiplier"

    # Configurable parameter for this plugin
    multiplier: int = 2

    @staticmethod
    def get_column_emoji() -> str:
        return "✖️"

    @property
    def required_columns(self) -> list[str]:
        """Columns that must exist before this generator runs."""
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        """Additional columns produced beyond the primary column."""
        return []
```

**Key points:**

- The `column_type` field must be a `Literal` type with a string default
- This value uniquely identifies your plugin (use kebab-case)
- Add any custom parameters your plugin needs (here: `multiplier`)
- `SingleColumnConfig` is a Pydantic model, so you can leverage all of Pydantic's validation features
- `get_column_emoji()` returns the emoji displayed in logs for this column type
- `required_columns` lists any columns this generator depends on (empty if none)
- `side_effect_columns` lists any additional columns this generator produces beyond the primary column (empty if none)

### Step 3: Create the implementation class

The implementation class defines the actual business logic of the plugin. For column generator plugins, inherit from `ColumnGeneratorFullColumn` or `ColumnGeneratorCellByCell` and implement the `generate` method.

Create `src/data_designer_index_multiplier/impl.py`:

```python
import logging

import pandas as pd
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

from data_designer_index_multiplier.config import IndexMultiplierColumnConfig

logger = logging.getLogger(__name__)


class IndexMultiplierColumnGenerator(ColumnGeneratorFullColumn[IndexMultiplierColumnConfig]):

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate the column data.

        Args:
            data: The current DataFrame being built

        Returns:
            The DataFrame with the new column added
        """
        logger.info(
            f"Generating column {self.config.name} "
            f"with multiplier {self.config.multiplier}"
        )

        data[self.config.name] = data.index * self.config.multiplier

        return data
```

**Key points:**

- Generic type `ColumnGeneratorFullColumn[IndexMultiplierColumnConfig]` connects the implementation to its config
- You have access to the configuration parameters via `self.config`

!!! info "Understanding generation_strategy"
    The `generation_strategy` specifies how the column generator will generate data. You choose a strategy by inheriting from the corresponding base class:

    - **`ColumnGeneratorFullColumn`**: Generates the full column (at the batch level) in a single call to `generate`
        - `generate` must take as input a `pd.DataFrame` with all previous columns and return a `pd.DataFrame` with the generated column appended.

    - **`ColumnGeneratorCellByCell`**: Generates one cell at a time
        - `generate` must take as input a `dict` with key/value pairs for all previous columns and return a `dict` with an additional key/value for the generated cell
        - Supports concurrent workers via a `max_parallel_requests` parameter on the configuration

### Step 4: Create the plugin object

Create a `Plugin` object that makes the plugin discoverable and connects the implementation and config classes.

Create `src/data_designer_index_multiplier/plugin.py`:

```python
from data_designer.plugins import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="data_designer_index_multiplier.config.IndexMultiplierColumnConfig",
    impl_qualified_name="data_designer_index_multiplier.impl.IndexMultiplierColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
```

### Step 5: Package your plugin

Create a `pyproject.toml` file to define your package and register the entry point:

```toml
[project]
name = "data-designer-index-multiplier"
version = "1.0.0"
description = "Data Designer index multiplier plugin"
requires-python = ">=3.10"
dependencies = [
    "data-designer",
]

# Register this plugin via entry points
[project.entry-points."data_designer.plugins"]
index-multiplier = "data_designer_index_multiplier.plugin:plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_designer_index_multiplier"]
```

!!! info "Entry Point Registration"
    Plugins are discovered automatically using [Python entry points](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata). It is important to register your plugin as an entry point under the `data_designer.plugins` group.

    The entry point format is:
    ```toml
    [project.entry-points."data_designer.plugins"]
    <entry-point-name> = "<module.path>:<plugin-instance-name>"
    ```

### Step 6: Use your plugin

Install your plugin in editable mode for testing:

```bash
# From the plugin directory
uv pip install -e .
```

Once installed, your plugin works just like built-in column types:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

from data_designer_index_multiplier.config import IndexMultiplierColumnConfig

data_designer = DataDesigner()
builder = dd.DataDesignerConfigBuilder()

# Add a regular column
builder.add_column(
    dd.SamplerColumnConfig(
        name="category",
        sampler_type="category",
        params=dd.CategorySamplerParams(values=["A", "B", "C"]),
    )
)

# Add your custom plugin column
builder.add_column(
    IndexMultiplierColumnConfig(
        name="scaled_index",
        multiplier=5,
    )
)

# Generate data
results = data_designer.create(builder, num_records=10)
print(results.load_dataset())
```

Output:
```
  category  scaled_index
0        B             0
1        A             5
2        C            10
3        A            15
4        B            20
...
```

---

## Validating Your Plugin

Data Designer provides a testing utility to validate that your plugin is structured correctly. Use `assert_valid_plugin` to check that your config and implementation classes are properly defined:

```python
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer_index_multiplier.plugin import plugin

# Raises AssertionError with a descriptive message if anything is wrong with the general plugin structure
assert_valid_plugin(plugin)
```

This validates that:

- The config class is a subclass of `ConfigBase`
- For column generator plugins: the implementation class is a subclass of `ConfigurableTask`
- For seed reader plugins: the implementation class is a subclass of `SeedReader`

---

## Multiple Plugins in One Package

A single Python package can register multiple plugins. Simply define multiple `Plugin` instances and register each one as a separate entry point:

```toml
[project.entry-points."data_designer.plugins"]
my-column-generator = "my_package.plugins.column_generator.plugin:column_generator_plugin"
my-seed-reader = "my_package.plugins.seed_reader.plugin:seed_reader_plugin"
```

For an example of this pattern, see the end-to-end test plugins in the [tests_e2e/](https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/tests_e2e) directory.

That's it! You now know how to create a Data Designer plugin. Package your plugin and share it with the community!
