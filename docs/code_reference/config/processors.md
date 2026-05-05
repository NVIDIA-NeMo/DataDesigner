# Processor Configurations

The `data_designer.config.processors` module defines declarative configuration objects for data transformations. These configs are added to a `DataDesignerConfig` or `DataDesignerConfigBuilder`; the engine later compiles them into runtime processor implementations.

!!! note "Config"
    This page documents the processor configs users add to a configuration. If you are implementing a processor plugin or changing processor execution, see [engine processors](../engine/processors.md).

For the plugin authoring workflow, see [Build Your Own](../../plugins/build_your_own.md).

::: data_designer.config.processors
