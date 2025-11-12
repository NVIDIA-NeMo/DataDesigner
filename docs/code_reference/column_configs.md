# Column Configurations

The `column_configs` module defines configuration objects for all Data Designer column types. Each configuration inherits from [SingleColumnConfig](#data_designer.config.column_configs.SingleColumnConfig), which provides shared arguments like the column `name`, whether to `drop` the column after generation, and the `column_type`.

!!! info "`column_type` is a discriminator field"
    The `column_type` argument is used to identify column types when deserializing the [Data Designer Config](data_designer_config.md) from JSON/YAML. It acts as the discriminator in a [discriminated union](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions), allowing Pydantic to automatically determine which column configuration class to instantiate.

::: data_designer.config.column_configs
