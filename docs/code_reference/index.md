# Code Reference

Data Designer is implemented as three installable packages that share the `data_designer` namespace. The packages are layered: user-facing interface code calls the engine, and the engine consumes declarative config objects.

| Package | Namespace | Role |
|---------|-----------|------|
| [`data-designer-config`](config/index.md) | `data_designer.config` | Configuration schemas, builder APIs, plugin registration objects, and result schemas. |
| [`data-designer-engine`](engine/index.md) | `data_designer.engine` | Runtime contracts and implementations for generation, seed reading, processing, and MCP tool execution. |
| [`data-designer`](interface/index.md) | `data_designer.interface` | Public entry points for previewing, creating, and inspecting generated datasets. |

The dependency direction is `interface -> engine -> config`. Config objects describe what should happen, engine objects implement how it happens, and interface objects expose the supported public API.
