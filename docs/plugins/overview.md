# Data Designer Plugins

!!! warning "Experimental Feature"
    The plugin system is currently **experimental** and under active development. The documentation, examples, and plugin interface are subject to significant changes in future releases. If you encounter any issues, have questions, or have ideas for improvement, please consider starting [a discussion on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/discussions).

## What are plugins?

Plugins are Python packages that extend Data Designer's capabilities without modifying the core library. Similar to [VS Code extensions](https://marketplace.visualstudio.com/vscode) and [Pytest plugins](https://docs.pytest.org/en/stable/reference/plugin_list.html), the plugin system empowers you to build specialized extensions for your specific use cases and share them with the community.

**Current capabilities**: Data Designer supports two plugin types:

- **Column Generator Plugins**: Custom column types you pass to the config builder's [add_column](../code_reference/config_builder.md#data_designer.config.config_builder.DataDesignerConfigBuilder.add_column) method.
- **Seed Reader Plugins**: Custom seed dataset readers that let you load data from new sources (e.g., databases, cloud storage, custom formats).

**Coming soon**: Plugin support for processors, validators, and more!

## How do you use plugins?

A Data Designer plugin is just a Python package configured with an [entry point](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata) that points to a Data Designer `Plugin` object. Using a plugin is as simple as installing the package:

```bash
# Install a local plugin (for development and testing)
uv pip install -e /path/to/your/plugin

# Or install a published plugin from PyPI
pip install data-designer-{plugin-name}
```

Once installed, plugins are automatically discovered and ready to use — no additional registration or configuration needed. See the [example plugin](example.md) for a complete walkthrough.

## How do you create plugins?

Creating a plugin involves three main steps:

### 1. Implement the Plugin Components

Each plugin has three components, and we recommend organizing them into separate files within a plugin subdirectory:

- **`config.py`** -- Configuration class defining user-facing parameters
    - Column generator plugins: inherit from `SingleColumnConfig` with a `column_type` discriminator
    - Seed reader plugins: inherit from `SeedSource` with a `seed_type` discriminator
- **`impl.py`** -- Implementation class containing the core logic
    - Column generator plugins: inherit from `ColumnGeneratorFullColumn` or `ColumnGeneratorCellByCell`
    - Seed reader plugins: inherit from `SeedReader`
- **`plugin.py`** -- A `Plugin` instance that connects the config and implementation classes

### 2. Package Your Plugin

- Set up a Python package with `pyproject.toml`
- Register your plugin using entry points under `data_designer.plugins`
- Define dependencies (including `data-designer`)

### 3. Install and Test Locally

- Install your plugin locally with `uv pip install -e .` (editable mode)
- No publishing required — your plugin is usable immediately after a local install
- Iterate on your plugin code with fast feedback

### 4. Share Your Plugin (Optional)

- Publish to PyPI or another package index to make it installable by anyone via `pip install`
- This step is only needed if you want others outside your environment to use the plugin

**Ready to get started?** See the [Example Plugin](example.md) for complete walkthroughs of both plugin types!
