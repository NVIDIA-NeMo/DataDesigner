# Sampler Parameters

Sampler parameter classes configure Data Designer's built-in samplers. Use them in [SamplerColumnConfig](column_configs.md#data_designer.config.column_configs.SamplerColumnConfig) to specify how sampled column values are generated.

!!! tip "Displaying available samplers and their parameters"
    The config builder has an `info` attribute that can be used to display the
    available sampler types and their parameters:
    ```python
    config_builder.info.display("samplers")
    ```

::: data_designer.config.sampler_params
