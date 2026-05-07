# Column Generators

Column generators execute column generation in the Data Designer engine. A generator receives the upstream data needed for its task, returns row or batch data with generated values added, and reports the generation strategy the scheduler should use.

Related pages: [column_configs](../config/column_configs.md), [Build Your Own](../../plugins/build_your_own.md), [Using Models in Plugins](../../plugins/models.md), and [Custom Columns](../../concepts/custom_columns.md).

## Configuration

User-facing column configs inherit from [SingleColumnConfig](../config/column_configs.md#data_designer.config.base.SingleColumnConfig) and define a unique `column_type` discriminator. During compilation, the engine may group related configs into multi-column configs for generators that create sampler or seed columns together.

## Generation strategy

Column generator base classes return [GenerationStrategy](../config/column_configs.md#data_designer.config.column_configs.GenerationStrategy) values to tell the engine whether they run per row or over a full batch.

## Implementation bases

Generators that operate on a full batch can inherit from [ColumnGeneratorFullColumn](#data_designer.engine.column_generators.generators.base.ColumnGeneratorFullColumn). Row-oriented non-model generators can inherit from [ColumnGeneratorCellByCell](#data_designer.engine.column_generators.generators.base.ColumnGeneratorCellByCell). Generators that create initial rows use [FromScratchColumnGenerator](#data_designer.engine.column_generators.generators.base.FromScratchColumnGenerator). Model-backed plugin generators should use [ColumnGeneratorWithModelRegistry](#data_designer.engine.column_generators.generators.base.ColumnGeneratorWithModelRegistry) or [ColumnGeneratorWithModel](#data_designer.engine.column_generators.generators.base.ColumnGeneratorWithModel); see [Using Models in Plugins](../../plugins/models.md) for authoring guidance.

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
