# Data Designer Plugins

Data Designer is built to be a general-purpose tool for synthetic data. The built-in column types, samplers, seed readers, and processors cover the common shapes, but real datasets often need bespoke generation logic, custom data sources, or domain-specific post-processing. Plugins are how that customization happens without forking the core library — they ship as installable Python packages and participate in Data Designer's config system and runtime just like the built-in objects.

## What are plugins?

Plugins are implemented and installed as Python packages. A single package can expose one plugin or a collection of related plugins. Use them when you want a shareable solution for bespoke generation, data loading, or processing.

## Supported plugin types

Data Designer supports three plugin types:

- **Column generator plugins**: Custom column types you pass to the config builder's [add_column](../code_reference/config_builder.md#data_designer.config.config_builder.DataDesignerConfigBuilder.add_column) method.
- **Seed reader plugins**: Custom seed dataset readers that load data from new sources, such as databases, cloud storage, or custom file formats.
- **Processor plugins**: Custom processors that transform data before batches, after batches, or after generation completes. Pass them to the config builder's [add_processor](../code_reference/config_builder.md#data_designer.config.config_builder.DataDesignerConfigBuilder.add_processor) method.

The [plugins code reference](../code_reference/plugins.md) documents the [Plugin](../code_reference/plugins.md#data_designer.plugins.plugin.Plugin) object and plugin-type enum. Column generator implementation APIs are documented separately in the [column generators code reference](../code_reference/generators.md).

## How do you use plugins?

Plugin packages register their `Plugin` objects through Python package [entry points](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata), so installing the package is all that's needed for Data Designer to pick it up:

```bash
# Install a local plugin for development and testing.
uv pip install -e /path/to/your/plugin

# Or install a published plugin from PyPI or another package index.
pip install data-designer-{plugin-name}
```

For implementation instructions across all plugin types, see [Build Your Own](implement.md). To find installable plugin packages, see the [Catalog](available.md). For deeper behavior details, see the [processor plugin guide](processor.md) or [FileSystemSeedReader plugins](filesystem_seed_reader.md).

## How do you create plugins?

### 1. Implement the plugin components

Each plugin has three components. We recommend organizing them into separate files within a plugin package:

- **`config.py`**: Configuration class defining user-facing parameters.
    - Column generator plugins inherit from [SingleColumnConfig](../code_reference/column_configs.md#data_designer.config.base.SingleColumnConfig) with a `column_type` discriminator.
    - Seed reader plugins inherit from `SeedSource` or `FileSystemSeedSource` with a `seed_type` discriminator.
    - Processor plugins inherit from `ProcessorConfig` with a `processor_type` discriminator.
- **`impl.py`**: Implementation class containing the core logic.
    - Column generator plugins inherit from [ColumnGeneratorFullColumn](../code_reference/generators.md#data_designer.engine.column_generators.generators.base.ColumnGeneratorFullColumn) or [ColumnGeneratorCellByCell](../code_reference/generators.md#data_designer.engine.column_generators.generators.base.ColumnGeneratorCellByCell).
    - Seed reader plugins inherit from `SeedReader`, or `FileSystemSeedReader` for directory-backed sources.
    - Processor plugins inherit from `Processor` and override callback methods such as `process_before_batch`, `process_after_batch`, or `process_after_generation`.
- **`plugin.py`**: A [Plugin](../code_reference/plugins.md#data_designer.plugins.plugin.Plugin) instance that connects the config and implementation classes.

### 2. Package your plugin

- Set up a Python package with `pyproject.toml`.
- Register your plugin using entry points under `data_designer.plugins`.
- Define dependencies, including `data-designer`.

### 3. Install and test locally

- Install your plugin locally with `uv pip install -e .` from the plugin package directory.
- No publishing is required. Your plugin is usable immediately after a local install.
- Iterate on your plugin code with fast feedback.

### 4. Share your plugin (optional)

Publish to PyPI or another package index if you want others outside your environment to install the plugin with `pip install`.

## Scaffolded first-party plugins

NVIDIA-maintained plugin packages live in the [DataDesignerPlugins](https://github.com/NVIDIA-NeMo/DataDesignerPlugins) repository. That repository includes a `ddp` CLI for scaffolding plugin packages, generating the plugin catalog, aggregating CODEOWNERS, and validating installed `data_designer.plugins` entry points.

## Discovery troubleshooting

If a plugin is installed but not available, check these items first:

- The entry point group must be exactly `data_designer.plugins`.
- Run editable installs from the plugin package directory: `uv pip install -e .`.
- `DISABLE_DATA_DESIGNER_PLUGINS=true` disables entry point discovery.
- Plugin import and load failures are logged as warnings during discovery.
- The plugin discriminator default must be a string. Use `column_type`, `seed_type`, or `processor_type`, depending on the plugin type.
- The discriminator default must convert to a valid enum key. For example, `regex-filter` becomes `REGEX_FILTER`.
- Avoid duplicate plugin names. Discovery stores plugins by `plugin.name`, which comes from the discriminator default.
- For column generator plugins, call `assert_valid_plugin` on your plugin object to catch common structural issues at import time.

## Next steps

- See [Build Your Own](implement.md) for column generator, seed reader, and processor plugin implementation patterns.
- See the [Catalog](available.md) for available first-party plugin packages and trusted install sources.
- See [Processor Plugins](processor.md) for a complete processor package example.
- See [FileSystemSeedReader Plugins](filesystem_seed_reader.md) for inline and packaged filesystem-backed seed reader examples.
- See the [Markdown Section Seed Reader recipe](../recipes/plugin_development/markdown_seed_reader.md) for a runnable single-file `1:N` filesystem reader example.
