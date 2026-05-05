# Interface Package

The `data-designer` package provides the top-level user-facing package surface. This section covers `data_designer.interface`, which contains `DataDesigner`, persisted dataset creation results, and interface-level errors.

This package sits above engine and config. `DataDesigner` accepts Data Designer configs, calls the runtime layer, and returns preview or persisted creation results.

Start with [DataDesigner](data_designer.md) for previewing, creating, and inspecting datasets from a config. Use [results](results.md) for the object returned by persisted dataset creation, and [errors](errors.md) for exceptions surfaced at the public API boundary.
