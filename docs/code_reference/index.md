# Code Reference

Data Designer is split across three installable packages that share the `data_designer` namespace. The package boundaries follow the main flow of the system: users declare a dataset, the runtime executes that declaration, and the public interface exposes supported entry points for user-facing workflows.

The dependency direction is `interface -> engine -> config`. Config objects remain declarative data, engine components provide imperative execution behavior, and interface objects define the supported public boundary.

- **[Config](config/index.md)** — the declarative package. It defines the contracts that describe what Data Designer should generate.
- **[Engine](engine/index.md)** — the runtime package. It implements the behavior that turns declarations into generated datasets.
- **[Interface](interface/index.md)** — the public entry-point package. It exposes the supported APIs for running Data Designer and working with results.
