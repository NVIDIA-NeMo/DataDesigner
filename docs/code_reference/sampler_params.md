# Sampler Parameters

The `sampler_params` module defines parameter configuration objects for all Data Designer sampler types. Sampler parameters are used within the [SamplerColumnConfig](column_configs.md#data_designer.config.column_configs.SamplerColumnConfig) to specify how values should be generated for sampled columns.

!!! tip "Displaying available samplers and their parameters"
    The config builder has an `info` attribute that can be used to display the
    available sampler types and their parameters:
    ```python
    config_builder.info.display("samplers")
    ```

::: data_designer.config.sampler_params
