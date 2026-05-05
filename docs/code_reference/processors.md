# Processors

The `processors` module defines configuration objects for post-generation data transformations. Processors run after column generation and can modify the dataset schema or content before output.

For the plugin authoring workflow, see [Build Your Own](../plugins/build_your_own.md).

## Configuration

::: data_designer.config.processors

## Implementation base

Plugin processors inherit from [Processor](#data_designer.engine.processing.processors.base.Processor) and override one or more callback methods (`process_before_batch`, `process_after_batch`, `process_after_generation`).

### `Processor` {#data_designer.engine.processing.processors.base.Processor}

::: data_designer.engine.processing.processors.base.Processor
    options:
      show_root_toc_entry: false
