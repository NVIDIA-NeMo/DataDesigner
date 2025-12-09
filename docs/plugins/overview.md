# Data Designer Plugins

!!! warning "Experimental Feature"
    The plugin system is currently **experimental** and under active development. The documentation, examples, and plugin interface are subject to significant changes in future releases. If you encounter any issues, have questions, or have ideas for improvement, please [open an issue on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/issues/new/choose).

## What are plugins?

Plugins are Python packages that extend Data Designer's capabilities without modifying the core library. Similar to [VS Code extensions](https://marketplace.visualstudio.com/vscode) and [Pytest plugins](https://docs.pytest.org/en/stable/reference/plugin_list.html), the plugin system empowers you to build specialized extensions for your specific use cases and share them with the community.

**Current capabilities**: Data Designer currently supports plugins for column generators (the column types you pass to the config builder's [add_column](../code_reference/config_builder.md#data_designer.config.config_builder.DataDesignerConfigBuilder.add_column) method).

**Coming soon**: Plugin support for processors, validators, and more!

## How do you use plugins?

A Data Designer plugin is just a Python package configured with an [entry point](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata) that points to a Data Designer `Plugin` object. Using a plugin is as simple as installing the package:

```bash
pip install data-designer-{plugin-name}
```

Once installed, plugins are automatically discovered and ready to use. See the [example plugin](example.md) for a complete walkthrough.

## How do you create plugins?

Creating a plugin involves three main steps:

### 1. Implement the Plugin Components

- Create a task class inheriting from `ColumnGenerator`
- Create a config class inheriting from `SingleColumnConfig`
- Instantiate a `Plugin` object connecting them

### 2. Package Your Plugin

- Set up a Python package with `pyproject.toml`
- Register your plugin using entry points
- Define dependencies (including `data-designer`)

### 3. Share Your Plugin

- Publish to PyPI or another package index
- Share with the community!

**Ready to get started?** See the [Example Plugin](example.md) for a complete walkthrough!
