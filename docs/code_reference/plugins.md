# Plugins

This page documents the public plugin registration objects. A plugin package registers a [Plugin](#data_designer.plugins.plugin.Plugin) object through an entry point in the `data_designer.plugins` group. This object ties a config class to its implementation class and declares its [PluginType](#data_designer.plugins.plugin.PluginType).

For the plugin authoring workflow, see [Build Your Own](../plugins/build_your_own.md). For implementation base classes, see [Column Generators](generators.md) and [Seed Readers](seed_readers.md).

## `Plugin` {#data_designer.plugins.plugin.Plugin}

::: data_designer.plugins.plugin.Plugin
    options:
      show_root_toc_entry: false

## `PluginType` {#data_designer.plugins.plugin.PluginType}

::: data_designer.plugins.plugin.PluginType
    options:
      show_root_toc_entry: false
