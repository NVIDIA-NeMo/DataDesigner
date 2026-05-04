# Processor Plugins

Processor plugins add custom dataset transformations that run before each batch, after each batch, or after generation completes. Use them when the transformation does not fit the column model, such as filtering seed rows, writing derived artifacts, or reshaping finished output.

This example builds a `regex-filter` processor that keeps rows whose column value matches a regular expression.

## Package structure

```text
data-designer-regex-filter/
|-- pyproject.toml
`-- src/
    `-- data_designer_regex_filter/
        |-- __init__.py
        |-- config.py
        |-- impl.py
        `-- plugin.py
```

## Create the config class

Processor plugin configs inherit from `ProcessorConfig` and define a `processor_type` discriminator with a unique string literal.

Create `src/data_designer_regex_filter/config.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import Field

from data_designer.config.base import ProcessorConfig


class RegexFilterProcessorConfig(ProcessorConfig):
    """Filters rows by regex pattern on a specified column."""

    processor_type: Literal["regex-filter"] = "regex-filter"
    column: str = Field(description="Column to match against.")
    pattern: str = Field(description="Regex pattern to match.")
    invert: bool = Field(default=False, description="If True, keep rows that do not match.")
```

The discriminator value becomes the plugin name Data Designer uses during discovery. Use a unique value, typically in kebab-case.

## Create the implementation class

Processor implementations inherit from `Processor[YourConfig]` and override one or more callbacks:

- `process_before_batch(...)`
- `process_after_batch(...)`
- `process_after_generation(...)`

Create `src/data_designer_regex_filter/impl.py`:

```python
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor

from data_designer_regex_filter.config import RegexFilterProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


class RegexFilterProcessor(Processor[RegexFilterProcessorConfig]):
    """Filters batch rows based on a regex pattern."""

    def process_before_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        compiled = re.compile(self.config.pattern)
        mask = data[self.config.column].astype(str).apply(lambda value: bool(compiled.search(value)))
        if self.config.invert:
            mask = ~mask
        return data[mask].reset_index(drop=True)
```

This implementation filters each batch after seed rows are loaded and before generated columns run.

!!! note "Execution mode"
    Row-count-changing pre-batch and post-batch processors work on the sync engine path. With `DATA_DESIGNER_ASYNC_ENGINE=1`, those stages currently reject row-count changes.

## Create the plugin object

Create `src/data_designer_regex_filter/plugin.py`:

```python
from __future__ import annotations

from data_designer.plugins import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="data_designer_regex_filter.config.RegexFilterProcessorConfig",
    impl_qualified_name="data_designer_regex_filter.impl.RegexFilterProcessor",
    plugin_type=PluginType.PROCESSOR,
)
```

The qualified names must point to classes that exist in importable modules.

## Register the entry point

Create `pyproject.toml`:

```toml
[project]
name = "data-designer-regex-filter"
version = "0.1.0"
description = "Data Designer regex filter processor plugin"
requires-python = ">=3.10"
dependencies = [
    "data-designer",
]

[project.entry-points."data_designer.plugins"]
regex-filter = "data_designer_regex_filter.plugin:plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_designer_regex_filter"]
```

Install it from the plugin package directory:

```bash
uv pip install -e .
```

The editable install registers the `data_designer.plugins` entry point so Data Designer can discover the processor.

## Use the processor

Once installed, import the config class and add it with `add_processor()`:

```python
import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer_regex_filter.config import RegexFilterProcessorConfig

seed_data = pd.DataFrame(
    {
        "category": ["keep", "drop", "keep", "drop"],
        "value": ["a", "b", "c", "d"],
    }
)

data_designer = DataDesigner()
builder = dd.DataDesignerConfigBuilder()
builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_data))
builder.add_column(
    dd.SamplerColumnConfig(
        name="label",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["example"]),
    )
)
builder.add_processor(
    RegexFilterProcessorConfig(
        name="keep_only",
        column="category",
        pattern="^keep$",
    )
)

preview = data_designer.preview(builder)
print(preview.dataset)
```

The generated dataset contains only rows where `category` matches `^keep$`.

## Callback selection

Choose the callback based on when your processor needs to see the data:

| Callback | Runs | Use when |
|----------|------|----------|
| `process_before_batch` | After seed rows are loaded, before generated columns run | You need to filter or reshape seed data before generation |
| `process_after_batch` | After each generated batch finishes | You need to drop, reshape, or write per-batch artifacts |
| `process_after_generation` | Once on the final combined dataset | You need global deduplication, aggregation, or final cleanup |

See [Processors](../concepts/processors.md) for the built-in processor model and [Plugins](overview.md) for discovery troubleshooting.
