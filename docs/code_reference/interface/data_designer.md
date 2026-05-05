# DataDesigner Interface

The `data_designer.interface.data_designer` module defines [DataDesigner](#data_designer.interface.data_designer.DataDesigner), the high-level interface for running Data Designer configurations. Users instantiate this class to validate a [DataDesignerConfigBuilder](../config/config_builder.md#data_designer.config.config_builder.DataDesignerConfigBuilder).

!!! note "Interface"
    This page documents the public boundary between declarative configuration and engine execution. Config objects declare the dataset shape; `DataDesigner` validates and executes those declarations through the engine.

For runtime settings passed through `set_run_config()`, see [run_config](../config/run_config.md). For persisted creation results returned by `create()`, see [results](results.md).

## `DataDesigner` {#data_designer.interface.data_designer.DataDesigner}

::: data_designer.interface.data_designer.DataDesigner
    options:
      show_root_toc_entry: false
