# Config Package

The `data-designer-config` package owns the `data_designer.config` namespace: the declarative layer users use to describe the dataset they want Data Designer to generate. It defines the configuration schemas, builder APIs, and related data models that describe generation inputs, runtime options, extensions, and result metadata.

Config objects declare dataset properties and generation requirements, but they do not execute generation directly. The engine consumes these objects and maps them to runtime behavior through registries.
