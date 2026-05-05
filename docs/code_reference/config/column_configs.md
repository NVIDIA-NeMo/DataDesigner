# Column Configurations

Column configs declare Data Designer's built-in column types. Each configuration inherits from [SingleColumnConfig](#data_designer.config.base.SingleColumnConfig), which provides shared arguments like the column `name`, whether to `drop` the column after generation, and the `column_type`.

For column generator implementation classes, see [column_generators](../engine/column_generators.md).

!!! info "`column_type` is a discriminator field"
    The `column_type` argument is used to identify column types when deserializing the [Data Designer Config](data_designer_config.md) from JSON/YAML. It acts as the discriminator in a [discriminated union](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions), allowing Pydantic to automatically determine which column configuration class to instantiate.

## `SingleColumnConfig` {#data_designer.config.base.SingleColumnConfig}

::: data_designer.config.base.SingleColumnConfig
    options:
      show_root_toc_entry: false

## Column configurations

::: data_designer.config.column_configs
