# Interface Errors

The `data_designer.interface.errors` module defines interface-level exceptions raised when generation, profiling, or buffer validation fails at the public API boundary. These errors inherit from `data_designer.errors.DataDesignerError`, allowing callers to catch either specific interface failures or the project-wide base error type.

The package-level `data_designer.interface` export lazily exposes [DataDesignerGenerationError](#data_designer.interface.errors.DataDesignerGenerationError) and [DataDesignerProfilingError](#data_designer.interface.errors.DataDesignerProfilingError). [InvalidBufferValueError](#data_designer.interface.errors.InvalidBufferValueError) is defined in this module.

## `DataDesignerGenerationError` {#data_designer.interface.errors.DataDesignerGenerationError}

::: data_designer.interface.errors.DataDesignerGenerationError
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
