# Engine Processor Implementations

The `data_designer.engine.processing.processors` modules implement runtime processor behavior. Processor configs live in [`data_designer.config.processors`](../config/processors.md); the engine processor registry maps those config types to the classes documented here.

Plugin processors inherit from [Processor](#data_designer.engine.processing.processors.base.Processor) and override one or more callback methods: `process_before_batch`, `process_after_batch`, or `process_after_generation`.

!!! note "Engine"
    This page is for Data Designer developers and processor plugin authors. For the user-facing processor config objects, see [processor configurations](../config/processors.md).

## Base Contract

### `Processor` {#data_designer.engine.processing.processors.base.Processor}

::: data_designer.engine.processing.processors.base.Processor
    options:
      show_root_toc_entry: false

## Built-In Implementations

### `DropColumnsProcessor` {#data_designer.engine.processing.processors.drop_columns.DropColumnsProcessor}

::: data_designer.engine.processing.processors.drop_columns.DropColumnsProcessor
    options:
      show_root_toc_entry: false

### `SchemaTransformProcessor` {#data_designer.engine.processing.processors.schema_transform.SchemaTransformProcessor}

::: data_designer.engine.processing.processors.schema_transform.SchemaTransformProcessor
    options:
      show_root_toc_entry: false

## Registry

### `ProcessorRegistry` {#data_designer.engine.processing.processors.registry.ProcessorRegistry}

::: data_designer.engine.processing.processors.registry.ProcessorRegistry
    options:
      show_root_toc_entry: false

### `create_default_processor_registry` {#data_designer.engine.processing.processors.registry.create_default_processor_registry}

::: data_designer.engine.processing.processors.registry.create_default_processor_registry
    options:
      show_root_toc_entry: false
