# Interface Package

The `data-designer` package owns the top-level user-facing package surface: `data_designer.interface`, `data_designer.cli`, and `data_designer.integrations`. In this code reference section, the focus is `data_designer.interface`, which provides the public `DataDesigner` entry point, persisted dataset creation results, and interface-level errors.

The interface layer is the boundary users call to validate configs, generate previews, create persisted datasets, inspect runtime information, load generated artifacts, and publish results through supported integrations.
