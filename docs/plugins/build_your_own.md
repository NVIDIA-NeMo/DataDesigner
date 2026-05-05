# Build Your Own

Data Designer supports three plugin types: **column generators**, **seed readers**, and **processors**. They all use the same package shape: a config class, an implementation class, and a `Plugin` object registered through a `data_designer.plugins` entry point.

Use this page as the implementation checklist for plugin packages. Each tab below shows the core files for one plugin type.

## Package shape

Use the same structure for each plugin package:

```text
data-designer-my-plugin/
|-- pyproject.toml
`-- src/
    `-- data_designer_my_plugin/
        |-- __init__.py
        |-- config.py
        |-- impl.py
        `-- plugin.py
```

## Implementation patterns

=== "Column generator"

    This `index-multiplier` plugin adds a custom column whose value is the row index multiplied by a configurable integer.

    !!! note "Model-backed generators"
        If your column generator interacts with models, include at least one `model_alias` field in the config and use the model registry from the implementation. See [Using Models in Plugins](models.md) for the registry access pattern.

    !!! info "Full-column vs cell-by-cell generators"
        The example below uses `ColumnGeneratorFullColumn` because it can fill the whole batch from the DataFrame index. Use `ColumnGeneratorCellByCell` when each row can be generated independently from its upstream values and your `generate` method should receive and return a row dictionary. Prefer `ColumnGeneratorFullColumn` for vectorized pandas operations, batched external APIs, or logic that needs to inspect or update the full batch at once.

    `config.py`:

    ```python
    from __future__ import annotations

    from typing import Literal

    from data_designer.config.base import SingleColumnConfig


    class IndexMultiplierColumnConfig(SingleColumnConfig):
        column_type: Literal["index-multiplier"] = "index-multiplier"
        multiplier: int = 2

        @staticmethod
        def get_column_emoji() -> str:
            return "✖️"

        @property
        def required_columns(self) -> list[str]:
            return []

        @property
        def side_effect_columns(self) -> list[str]:
            return []
    ```

    `impl.py`:

    ```python
    from __future__ import annotations

    from typing import TYPE_CHECKING

    from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn

    from data_designer_index_multiplier.config import IndexMultiplierColumnConfig

    if TYPE_CHECKING:
        import pandas as pd


    class IndexMultiplierColumnGenerator(ColumnGeneratorFullColumn[IndexMultiplierColumnConfig]):
        def generate(self, data: pd.DataFrame) -> pd.DataFrame:
            data[self.config.name] = data.index * self.config.multiplier
            return data
    ```

    `plugin.py`:

    ```python
    from __future__ import annotations

    from data_designer.plugins import Plugin, PluginType

    plugin = Plugin(
        config_qualified_name="data_designer_index_multiplier.config.IndexMultiplierColumnConfig",
        impl_qualified_name="data_designer_index_multiplier.impl.IndexMultiplierColumnGenerator",
        plugin_type=PluginType.COLUMN_GENERATOR,
    )
    ```

    Entry point:

    ```toml
    [project.entry-points."data_designer.plugins"]
    index-multiplier = "data_designer_index_multiplier.plugin:plugin"
    ```

    For the generator implementation contract, see [Column Generators](../code_reference/engine/column_generators.md). For inline custom functions, see [Custom Columns](../concepts/custom_columns.md).

=== "Seed reader"

    This `prefixed-text-files` plugin loads text files from a directory and emits a seed dataset with prefixed file contents.

    `config.py`:

    ```python
    from __future__ import annotations

    from typing import Literal

    from data_designer.config.seed_source import FileSystemSeedSource


    class PrefixedTextSeedSource(FileSystemSeedSource):
        seed_type: Literal["prefixed-text-files"] = "prefixed-text-files"
        prefix: str = "plugin"
    ```

    `impl.py`:

    ```python
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    import data_designer.lazy_heavy_imports as lazy
    from data_designer.engine.resources.seed_reader import (
        FileSystemSeedReader,
        SeedReaderFileSystemContext,
    )

    from data_designer_prefixed_text_seed_reader.config import PrefixedTextSeedSource


    class PrefixedTextSeedReader(FileSystemSeedReader[PrefixedTextSeedSource]):
        output_columns = ["relative_path", "file_name", "prefixed_content"]

        def build_manifest(
            self,
            *,
            context: SeedReaderFileSystemContext,
        ) -> lazy.pd.DataFrame | list[dict[str, str]]:
            matched_paths = self.get_matching_relative_paths(
                context=context,
                file_pattern=self.source.file_pattern,
                recursive=self.source.recursive,
            )
            return [
                {
                    "relative_path": relative_path,
                    "file_name": Path(relative_path).name,
                }
                for relative_path in matched_paths
            ]

        def hydrate_row(
            self,
            *,
            manifest_row: dict[str, Any],
            context: SeedReaderFileSystemContext,
        ) -> dict[str, str]:
            relative_path = str(manifest_row["relative_path"])
            with context.fs.open(relative_path, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
            return {
                "relative_path": relative_path,
                "file_name": str(manifest_row["file_name"]),
                "prefixed_content": f"{self.source.prefix}:{content}",
            }
    ```

    `plugin.py`:

    ```python
    from __future__ import annotations

    from data_designer.plugins import Plugin, PluginType

    plugin = Plugin(
        config_qualified_name="data_designer_prefixed_text_seed_reader.config.PrefixedTextSeedSource",
        impl_qualified_name="data_designer_prefixed_text_seed_reader.impl.PrefixedTextSeedReader",
        plugin_type=PluginType.SEED_READER,
    )
    ```

    Entry point:

    ```toml
    [project.entry-points."data_designer.plugins"]
    prefixed-text-files = "data_designer_prefixed_text_seed_reader.plugin:plugin"
    ```

    For the engine API behind this example, see [Seed Readers](../code_reference/engine/seed_readers.md).

=== "Processor"

    This `regex-filter` plugin filters rows whose column value matches a regular expression.

    `config.py`:

    ```python
    from __future__ import annotations

    from typing import Literal

    from pydantic import Field

    from data_designer.config.base import ProcessorConfig


    class RegexFilterProcessorConfig(ProcessorConfig):
        processor_type: Literal["regex-filter"] = "regex-filter"
        column: str = Field(description="Column to match against.")
        pattern: str = Field(description="Regex pattern to match.")
        invert: bool = Field(default=False, description="If True, keep rows that do not match.")
    ```

    `impl.py`:

    ```python
    from __future__ import annotations

    from typing import TYPE_CHECKING

    from data_designer.engine.processing.processors.base import Processor

    from data_designer_regex_filter.config import RegexFilterProcessorConfig

    if TYPE_CHECKING:
        import pandas as pd


    class RegexFilterProcessor(Processor[RegexFilterProcessorConfig]):
        def process_before_batch(self, data: pd.DataFrame) -> pd.DataFrame:
            mask = data[self.config.column].astype(str).str.contains(self.config.pattern, regex=True)
            if self.config.invert:
                mask = ~mask
            return data[mask].reset_index(drop=True)
    ```

    `plugin.py`:

    ```python
    from __future__ import annotations

    from data_designer.plugins import Plugin, PluginType

    plugin = Plugin(
        config_qualified_name="data_designer_regex_filter.config.RegexFilterProcessorConfig",
        impl_qualified_name="data_designer_regex_filter.impl.RegexFilterProcessor",
        plugin_type=PluginType.PROCESSOR,
    )
    ```

    Entry point:

    ```toml
    [project.entry-points."data_designer.plugins"]
    regex-filter = "data_designer_regex_filter.plugin:plugin"
    ```

    For callback selection and processor execution details, see [Processors](../concepts/processors.md). For the engine API behind this example, see [Engine Processors code reference](../code_reference/engine/processors.md).

## Install and use locally

Install any plugin package in editable mode from the package directory:

```bash
uv pip install -e .
```

The editable install registers the `data_designer.plugins` entry point so Data Designer can discover the plugin.

!!! note "Restart your kernel after installing"
    Data Designer caches the plugin registry on first import, so an `import data_designer` that already happened in your Python process — typical in a notebook — won't pick up a freshly installed plugin. After `uv pip install -e .`, restart the kernel (or interpreter) so the next import rebuilds the registry.

## Validate plugins

Data Designer provides a testing utility for common plugin structure checks:

```python
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer_index_multiplier.plugin import plugin

assert_valid_plugin(plugin)
```

`assert_valid_plugin` checks that the plugin's config inherits from `ConfigBase` and that the implementation class inherits from the appropriate base for its plugin type (`ConfigurableTask` for column generators, `SeedReader` for seed readers).

For published plugins, add at least one functional test that runs the plugin through `DataDesigner.preview(...)`. This catches packaging and entry point issues that a direct implementation test can miss.

## Multiple plugins in one package

A single Python package can register multiple plugins by defining multiple `Plugin` objects and entry points:

```toml
[project.entry-points."data_designer.plugins"]
my-column-generator = "my_package.plugins.column_generator.plugin:column_generator_plugin"
my-seed-reader = "my_package.plugins.seed_reader.plugin:seed_reader_plugin"
my-processor = "my_package.plugins.processor.plugin:processor_plugin"
```
