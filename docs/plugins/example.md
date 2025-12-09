!!! warning "Experimental Feature"
    The plugin system is currently **experimental** and under active development. The documentation, examples, and plugin interface are subject to significant changes in future releases. If you encounter any issues, have questions, or have ideas for improvement, please [open an issue on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/issues/new/choose).


# Example Plugin: Index Multiplier

In this guide, we will build a simple plugin that generates values by multiplying the row index by a user-specified multiplier. Admittedly, not the most useful plugin, but it demonstrates the required steps 😜.

A Data Designer plugin is implemented as a Python package with three main components:

1. **Configuration Class**: Defines the parameters users can configure
2. **Task Class**: Contains the core implementation of the plugin
3. **Plugin Object**: Connects the config and task classes to make the plugin discoverable

Let's build the `data-designer-index-multiplier` plugin step by step.

## Step 1: Create a Python package

Data Designer plugins are implemented as Python packages. We recommend using a standard structure for your plugin package.

For example, here is the structure of a `data-designer-index-multiplier` plugin:

```
data-designer-index-multiplier/
├── pyproject.toml
└── src/
    └── data_designer_index_multiplier/
        ├── __init__.py
        └── plugin.py
```

## Step 2: Create the config class

The configuration class defines what parameters users can set when using your plugin. For column generator plugins, it must inherit from [SingleColumnConfig](../code_reference/column_configs.md#data_designer.config.column_configs.SingleColumnConfig) and include a [discriminator field](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions).

```python
from typing import Literal
from data_designer.config.column_configs import SingleColumnConfig

class IndexMultiplierColumnConfig(SingleColumnConfig):
    """Configuration for the index multiplier column generator."""

    # Configurable parameter for this plugin
    multiplier: int = 2

    # Required: discriminator field with a unique Literal type
    # This value identifies your plugin and becomes its column_type
    column_type: Literal["index-multiplier"] = "index-multiplier"
```

**Key points:**

- The `column_type` field must be a `Literal` type with a string default
- This value uniquely identifies your plugin (use kebab-case)
- Add any custom parameters your plugin needs (here: `multiplier`)
- `SingleColumnConfig` is a Pydantic model, so you can leverage all of Pydantic's validation features

## Step 3: Create the task class

The task class implements the actual business logic of the plugin. For column generator plugins, it inherits from [ColumnGenerator](../code_reference/column_generators.md#data_designer.engine.column_generators.generators.base.ColumnGenerator) and must implement a `metadata` static method and `generate` method:


```python
import logging
import pandas as pd

from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)

# Data Designer uses the standard Python logging module for logging
logger = logging.getLogger(__name__)

class IndexMultiplierColumnGenerator(ColumnGenerator[IndexMultiplierColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        """Define metadata about this generator."""
        return GeneratorMetadata(
            name="index-multiplier",
            description="Generates values by multiplying the row index by a user-specified multiplier",
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            required_resources=None,
        )

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

        # Access config via self.config
        data[self.config.name] = data.index * self.config.multiplier

        return data
```

**Key points:**

- Generic type `ColumnGenerator[IndexMultiplierColumnConfig]` connects the task to its config
- `metadata()` describes your generator and its requirements
- `generation_strategy` can be `FULL_COLUMN`, `ROW_WISE`, or `BATCH`
- `required_resources` lists any required resources (models, artifact storage, etc.). This parameter will change in the future, so keeping it as `None` is safe for now.
- Access configuration parameters via `self.config`

!!! info "Understanding generation_strategy"
    The `generation_strategy` specifies how the column generator will generate data.

    - **`FULL_COLUMN`**: Generates the entire column at once
        - `generate` must take a `pd.DataFrame` as input and return a `pd.DataFrame`

    - **`CELL_BY_CELL`**: Generates one cell at a time
        - `generate` must take a `dict` as input and return a `dict`
        - Supports concurrent workers via a `max_parallel_requests` parameter on the configuration

## Step 4: Create the plugin object

Create a `Plugin` object that makes the plugin discoverable and connects the task and config classes.

```python
from data_designer.plugins import Plugin, PluginType

# Plugin instance - this is what gets loaded via entry point
plugin = Plugin(
    task_cls=IndexMultiplierColumnGenerator,
    config_cls=IndexMultiplierColumnConfig,
    plugin_type=PluginType.COLUMN_GENERATOR,
    emoji="🔌",
)
```

### Complete plugin code

Pulling it all together, here is the complete plugin code for `src/data_designer_index_multiplier/plugin.py`:

```python
import logging
from typing import Literal

import pandas as pd

from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer.plugins import Plugin, PluginType

# Data Designer uses the standard Python logging module for logging
logger = logging.getLogger(__name__)


class IndexMultiplierColumnConfig(SingleColumnConfig):
    """Configuration for the index multiplier column generator."""

    # Configurable parameter for this plugin
    multiplier: int = 2

    # Required: discriminator field with a unique Literal type
    # This value identifies your plugin and becomes its column_type
    column_type: Literal["index-multiplier"] = "index-multiplier"


class IndexMultiplierColumnGenerator(ColumnGenerator[IndexMultiplierColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        """Define metadata about this generator."""
        return GeneratorMetadata(
            name="index-multiplier",
            description="Generates values by multiplying the row index by a user-specified multiplier",
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            required_resources=None,
        )

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

        # Access config via self.config
        data[self.config.name] = data.index * self.config.multiplier

        return data


# Plugin instance - this is what gets loaded via entry point
plugin = Plugin(
    task_cls=IndexMultiplierColumnGenerator,
    config_cls=IndexMultiplierColumnConfig,
    plugin_type=PluginType.COLUMN_GENERATOR,
    emoji="🔌",
)
```

## Step 5: Package your plugin

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

## Step 6: Use your plugin

Install your plugin in editable mode for testing:

```bash
# From the plugin directory
uv pip install -e .
```

Once installed, your plugin works just like built-in column types:

```python
from data_designer_index_multiplier.plugin import IndexMultiplierColumnConfig

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    SamplerColumnConfig,
)

data_designer = DataDesigner()
builder = DataDesignerConfigBuilder()

# Add a regular column
builder.add_column(
    SamplerColumnConfig(
        name="category",
        sampler_type="category",
        params=CategorySamplerParams(values=["A", "B", "C"]),
    )
)

# Add your custom plugin column
builder.add_column(
    IndexMultiplierColumnConfig(
        name="v",
        multiplier=5,
    )
)

# Generate data
results = data_designer.create(builder, num_records=10)
print(results.load_dataset())
```

Output:
```
  category  multiplied-index
0        B                 0
1        A                 5
2        C                10
3        A                15
4        B                20
...
```

That's it! You have now created and used your first Data Designer plugin. The last step is to package your plugin and share it with the community 🚀
