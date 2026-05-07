# Config Package

The `data-designer-config` package provides `data_designer.config`, the configuration layer of Data Designer. It contains the objects used to describe dataset structure, model access, tool access, seed data, sampler parameters, validators, processors, run settings, plugin registrations, and analysis results.

This package is the base of the dependency chain. Engine and interface code consume these config objects, but config objects do not execute generation directly.

For programmatic configuration work, start with [config_builder](config_builder.md) and [data_designer_config](data_designer_config.md). Use the narrower pages for exact constructor fields for columns, models, MCP tools, seeds, processors, samplers, validators, or profiling results.
