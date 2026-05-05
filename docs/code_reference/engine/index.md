# Engine Package

The `data-designer-engine` package owns the `data_designer.engine` namespace: the runtime layer that turns declarative Data Designer configs into generated datasets. It coordinates execution, runtime resources, model access, plugin behavior, validation, post-processing, and analysis behind the public config and interface APIs.

This section is primarily for Data Designer developers and plugin authors. Most end users interact with the engine indirectly through `data_designer.config` and `data_designer.interface`.
