# Column Configurations

The `column_configs` module defines configuration classes for all DataDesigner column types. Each configuration class inherits from [SingleColumnConfig](#data_designer.config.column_configs.SingleColumnConfig), which provides the shared configuration options like the column `name`, whether to `drop` the column after generation, and the `column_type` (which is used as the discriminator in a [discriminated union](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions) when working with serialized configurations) .

::: data_designer.config.column_configs
