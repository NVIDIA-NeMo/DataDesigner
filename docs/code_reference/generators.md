# Column Generators

This page documents the base classes that execute column generation in the Data Designer engine. A column generator receives upstream data, writes its configured output column and any side-effect columns, and reports the generation strategy the scheduler should use.

Built-in generators and custom extension generators use the same implementation contract. Plugin packages can register a custom generator implementation with a [Plugin](plugins.md#data_designer.plugins.plugin.Plugin) object; for that packaging flow, see [Build Your Own](../plugins/implement.md). For inline user-defined functions, see [Custom Columns](../concepts/custom_columns.md).

## Configuration

Column generator configs inherit from [SingleColumnConfig](column_configs.md#data_designer.config.base.SingleColumnConfig) and define a unique `column_type` discriminator. See [Column Configurations](column_configs.md) for the config-side API.

## Generation strategy

Column generator base classes return [GenerationStrategy](column_configs.md#data_designer.config.column_configs.GenerationStrategy) values to tell the engine whether they run per row or over a full batch.

## Implementation bases

Most generators inherit from [ColumnGeneratorFullColumn](#data_designer.engine.column_generators.generators.base.ColumnGeneratorFullColumn) or [ColumnGeneratorCellByCell](#data_designer.engine.column_generators.generators.base.ColumnGeneratorCellByCell), depending on whether they operate on a full batch or one row at a time.

### `ColumnGenerator` {#data_designer.engine.column_generators.generators.base.ColumnGenerator}

::: data_designer.engine.column_generators.generators.base.ColumnGenerator
    options:
      show_root_toc_entry: false

### `ColumnGeneratorFullColumn` {#data_designer.engine.column_generators.generators.base.ColumnGeneratorFullColumn}

::: data_designer.engine.column_generators.generators.base.ColumnGeneratorFullColumn
    options:
      show_root_toc_entry: false

### `ColumnGeneratorCellByCell` {#data_designer.engine.column_generators.generators.base.ColumnGeneratorCellByCell}

::: data_designer.engine.column_generators.generators.base.ColumnGeneratorCellByCell
    options:
      show_root_toc_entry: false

### `FromScratchColumnGenerator` {#data_designer.engine.column_generators.generators.base.FromScratchColumnGenerator}

::: data_designer.engine.column_generators.generators.base.FromScratchColumnGenerator
    options:
      show_root_toc_entry: false

### `ColumnGeneratorWithModelRegistry` {#data_designer.engine.column_generators.generators.base.ColumnGeneratorWithModelRegistry}

::: data_designer.engine.column_generators.generators.base.ColumnGeneratorWithModelRegistry
    options:
      show_root_toc_entry: false

### `ColumnGeneratorWithModel` {#data_designer.engine.column_generators.generators.base.ColumnGeneratorWithModel}

::: data_designer.engine.column_generators.generators.base.ColumnGeneratorWithModel
    options:
      show_root_toc_entry: false
