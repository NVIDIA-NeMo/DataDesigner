# Data Designer Plugins

Plugins let you add new object types to Data Designer without modifying the core library. Once installed, plugins behave like native Data Designer objects: they use the same declarative config patterns, builder APIs, discovery flow, and runtime execution paths as the built-in objects.

## Supported plugin types

Data Designer supports three plugin types:

- **Column generator plugins**: Custom [column generators](../code_reference/engine/column_generators.md) you pass to the config builder's [add_column](../code_reference/config/config_builder.md#data_designer.config.config_builder.DataDesignerConfigBuilder.add_column) method.
- **Seed reader plugins**: Custom [seed readers](../code_reference/engine/seed_readers.md) that load data from new sources, such as databases, cloud storage, or custom file formats.
- **Processor plugins**: Custom [processor implementations](../code_reference/engine/processors.md) configured by processor config objects that transform data before batches, after batches, or after generation completes. Pass them to the config builder's [add_processor](../code_reference/config/config_builder.md#data_designer.config.config_builder.DataDesignerConfigBuilder.add_processor) method.

## Use an Installed Plugin

Plugin packages register their `Plugin` objects through Python package [entry points](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata). Data Designer discovers installed plugin entry points automatically, so no extra registration code is required. Once a plugin package is installed, use its new object types in your Data Designer workflow.

If you install a plugin after `data_designer` has already been imported, restart the Python process so plugin discovery can rebuild from the new entry points.

## Build a Plugin

For implementation instructions across all plugin types, see the [Build Your Own](build_your_own.md) section.

## Find Plugins

Use the Data Designer CLI to discover and install published plugin packages from catalogs. See [Discover Plugins](discover.md) for the catalog workflow, first-party plugin documentation, and source links.

## Discovery troubleshooting

If a plugin is installed but not available, check these items first:

- The entry point group must be exactly `data_designer.plugins`.
- Check the value of the `DISABLE_DATA_DESIGNER_PLUGINS` environment variable. If it is set to `true`, entry point discovery is disabled.
- The plugin discriminator default must be a string. Use `column_type`, `seed_type`, or `processor_type`, depending on the plugin type.
- Avoid duplicate plugin names. Discovery stores plugins by `plugin.name`, which comes from the discriminator default.
- For plugin packages under development, call `assert_valid_plugin` on the plugin object to catch common structural issues at import time.
