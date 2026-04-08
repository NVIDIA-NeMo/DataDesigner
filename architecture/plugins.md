# Plugins

The plugin system allows third-party packages to extend DataDesigner with new column types, seed source types, and processor types. Plugins are discovered via Python entry points and injected into the config layer's discriminated unions at import time.

Source: `packages/data-designer-config/src/data_designer/plugins/` and `packages/data-designer-config/src/data_designer/plugin_manager.py`

## Overview

DataDesigner's type system is built on Pydantic discriminated unions. The plugin system extends these unions at runtime so that:
- User configs can reference plugin-provided types by name
- Pydantic validation and deserialization work correctly for plugin types
- The engine's registries can dispatch to plugin-provided generators

Plugins are standard Python packages that declare entry points in the `data_designer.plugins` group.

## Key Components

### Plugin Descriptor

`Plugin` (in `plugins/plugin.py`) is a Pydantic model describing a plugin:
- **`impl_qualified_name`** — fully qualified name of the implementation class (e.g., generator)
- **`config_qualified_name`** — fully qualified name of the config class
- **`PluginType`** — one of `COLUMN_GENERATOR`, `SEED_READER`, or `PROCESSOR`

Validators ensure:
- Both modules exist and are importable
- The config class has the correct `Literal` discriminator field (`column_type`, `seed_type`, or `processor_type` depending on plugin type)
- The plugin `name` is derived from the discriminator field's default value

### PluginRegistry

Singleton (`__new__` + class-level `_instance`) that scans `importlib.metadata.entry_points(group="data_designer.plugins")` on first construction. Each entry point is loaded and expected to return a `Plugin` instance.

Plugins can be disabled globally with `DISABLE_DATA_DESIGNER_PLUGINS=true`.

### PluginManager

Thin facade over `PluginRegistry` providing typed injection methods:
- `inject_into_column_config_type_union` — extends `ColumnConfigT`
- `inject_into_seed_source_type_union` — extends `SeedSourceT`
- `inject_into_processor_config_type_union` — extends `ProcessorConfigT`

Each method ORs the plugin's config class into the existing type union (`type_union |= plugin.config_cls`).

### Integration Points

Plugin injection happens at module load time in the config layer:
- `column_types.py` instantiates `PluginManager()` and extends `ColumnConfigT`
- `seed_source_types.py` extends `SeedSourceT`
- `processor_types.py` extends `ProcessorConfigT`

On the engine side, `create_default_column_generator_registry()` merges plugin entry points into the `ColumnGeneratorRegistry`, mapping plugin column types to their generator implementations.

## Data Flow

### Discovery (at import time)
```
import data_designer.config.column_types
  → PluginManager() → PluginRegistry()
  → scan entry_points(group="data_designer.plugins")
  → load each entry point → Plugin instance
  → inject_into_column_config_type_union
  → ColumnConfigT now includes plugin config classes
```

### Usage (at runtime)
```
User config includes column_type: "my-plugin-type"
  → Pydantic discriminated union matches plugin config class
  → DatasetBuilder looks up generator in ColumnGeneratorRegistry
  → Plugin's generator class handles generation
```

### Relationship to Custom Columns

Plugins and custom columns serve different use cases:

| | Entry-Point Plugins | Custom Columns (`@custom_column_generator`) |
|---|---|---|
| **Scope** | Installable packages, new column types | In-process callables, same session |
| **Discovery** | `importlib.metadata.entry_points` | Decorator attaches metadata to callable |
| **Type system** | New `column_type` discriminator value | Uses built-in `custom` column type |
| **Distribution** | pip-installable | Code in the user's script/notebook |

## Design Decisions

- **Entry points over explicit registration** — plugins are discovered automatically when installed. Users don't need to modify DataDesigner configs or code to activate a plugin; `pip install` is sufficient.
- **Union injection at import time** ensures Pydantic validation works for plugin types without any runtime setup. The tradeoff is that plugin discovery runs on first import of the config layer.
- **`DISABLE_DATA_DESIGNER_PLUGINS`** provides an escape hatch for environments where plugin loading is undesirable (testing, CI, restricted environments).
- **Singleton registry** prevents duplicate plugin scanning when multiple modules import the config layer.

## Cross-References

- [System Architecture](overview.md) — where plugins fit in the stack
- [Config Layer](config.md) — discriminated unions that plugins extend
- [Engine Layer](engine.md) — generator registry that plugins populate
- [Agent Introspection](agent-introspection.md) — how plugin types appear in type discovery
