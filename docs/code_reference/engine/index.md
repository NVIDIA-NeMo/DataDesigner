# Engine Package

The `data-designer-engine` package provides `data_designer.engine`, the runtime layer of Data Designer. It consumes `data_designer.config` objects and maps them to execution behavior through generators, seed readers, processors, registries, model access, and MCP tool execution.

This package sits between config and interface: it depends on config, and the public interface calls into it. Use these pages for plugin implementation contracts, registry behavior, seed reader internals, processor execution, column generator bases, and MCP runtime behavior.
