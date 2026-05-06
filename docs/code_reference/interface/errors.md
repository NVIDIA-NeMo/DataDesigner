# Interface Errors

Interface errors represent failures surfaced at the public API boundary. DataDesignerGenerationError wraps dataset generation failures from `create()` and `preview()`, DataDesignerEarlyShutdownError identifies generation runs that terminate early without producing records, and DataDesignerProfilingError wraps profiling failures from those methods. These errors inherit from `data_designer.errors.DataDesignerError`, allowing callers to catch either specific interface failures or the project-wide base error type.

The package-level `data_designer.interface` export lazily exposes [DataDesignerGenerationError](#data_designer.interface.errors.DataDesignerGenerationError), [DataDesignerEarlyShutdownError](#data_designer.interface.errors.DataDesignerEarlyShutdownError), and [DataDesignerProfilingError](#data_designer.interface.errors.DataDesignerProfilingError). [InvalidBufferValueError](#data_designer.interface.errors.InvalidBufferValueError) is defined in this module.

## `DataDesignerGenerationError` {#data_designer.interface.errors.DataDesignerGenerationError}

::: data_designer.interface.errors.DataDesignerGenerationError
    options:
      show_root_toc_entry: false

## `DataDesignerEarlyShutdownError` {#data_designer.interface.errors.DataDesignerEarlyShutdownError}

::: data_designer.interface.errors.DataDesignerEarlyShutdownError
    options:
      show_root_toc_entry: false

## `DataDesignerProfilingError` {#data_designer.interface.errors.DataDesignerProfilingError}

::: data_designer.interface.errors.DataDesignerProfilingError
    options:
      show_root_toc_entry: false

## `InvalidBufferValueError` {#data_designer.interface.errors.InvalidBufferValueError}

::: data_designer.interface.errors.InvalidBufferValueError
    options:
      show_root_toc_entry: false
