# Config Layer

The config layer (`data_designer.config`) defines the declarative surface of DataDesigner. Users describe what their data should look like; the config layer validates and structures those declarations. It never calls the engine directly.

Source: `packages/data-designer-config/src/data_designer/config/`

## Overview

The config layer provides:
- **`DataDesignerConfigBuilder`** — fluent builder for constructing dataset configs
- **`DataDesignerConfig`** — the root config object holding columns, models, constraints, processors, and profilers
- **Column configs** — a discriminated union of Pydantic models, one per column type
- **Model configs** — LLM endpoint configuration with inference parameters
- **Sampler params** — statistical generator parameters with their own discriminated union
- **Plugin integration** — runtime extension of config unions via entry-point plugins

## Key Components

### Builder API

`DataDesignerConfigBuilder` is the primary construction surface. It holds mutable state (column configs, constraints, processors) and produces an immutable `DataDesignerConfig` on `build()`.

- **Fluent mutators**: `add_column`, `add_constraint`, `add_processor`, `add_profiler`, `add_model_config`, `add_tool_config`, `with_seed_dataset`
- **Column shorthand**: pass `name` + `column_type` + kwargs instead of a full config instance; the builder resolves the correct config class via `get_column_config_from_kwargs`
- **Config loading**: `from_config` accepts dicts, file paths, URLs, or `BuilderConfig` objects; normalizes shorthand formats into the full structure

`BuilderConfig` wraps `DataDesignerConfig` with a `library_version` field validated against the running version.

### Column Configs

All column configs inherit from `SingleColumnConfig(ConfigBase, ABC)`, which provides `name`, `drop`, `allow_resize`, and the `column_type` discriminator field.

Concrete types include: `SamplerColumnConfig`, `LLMTextColumnConfig`, `LLMStructuredColumnConfig`, `LLMCodeColumnConfig`, `LLMJudgeColumnConfig`, `EmbeddingColumnConfig`, `ImageColumnConfig`, `ValidationColumnConfig`, `ExpressionColumnConfig`, `SeedDatasetColumnConfig`, `CustomColumnConfig`.

Each fixes `column_type: Literal["..."]` with a kebab-case string. The full union `ColumnConfigT` is built at module load time and extended by plugins.

### Discriminated Unions

Pydantic discriminated unions are the backbone of config deserialization:

- **`DataDesignerConfig.columns`**: `list[Annotated[ColumnConfigT, Field(discriminator="column_type")]]` — picks the right config class from the `column_type` field
- **`SamplerColumnConfig.params`**: `Annotated[SamplerParamsT, Discriminator("sampler_type")]` — nested discrimination for sampler parameters
- **`InferenceParamsT`**: discriminated on `generation_type` (chat completion, embedding, image)

A `model_validator(mode="before")` on `SamplerColumnConfig` injects `sampler_type` into nested param dicts when users omit it, enabling a cleaner shorthand.

### Model Configs

`ModelConfig` holds `alias`, `model`, `inference_parameters` (discriminated), optional `provider`, and `skip_health_check`. Inference parameters support distribution-valued fields (`temperature`, `top_p` can be `UniformDistribution` or `ManualDistribution` with a `sample()` method).

`ModelProvider` configures the endpoint: URL, provider type (default `openai`), auth, headers, extra body parameters.

### ConfigBase

`ConfigBase` is the shared Pydantic base: `extra="forbid"`, enums serialized as values. It must not import other `data_designer.*` modules to keep it as a minimal dependency island.

## Data Flow

1. User calls builder methods or loads YAML/JSON
2. Builder resolves column type → config class via `get_column_config_cls_from_type` (built-in map, then plugin fallback)
3. For sampler columns, `_resolve_sampler_kwargs` maps `sampler_type` → params class via `SAMPLER_PARAMS`
4. `build()` triggers Pydantic validation on the full `DataDesignerConfig`
5. The validated config is passed to the engine for compilation and execution

## Design Decisions

- **Config objects are data, not behavior.** They define structure and constraints but never call the engine. This keeps the dependency direction clean (engine depends on config, not the reverse).
- **Discriminated unions over class hierarchies** for column types. Pydantic handles deserialization dispatch; adding a new type means adding a config class with the right `Literal` discriminator, not modifying a factory.
- **Plugin injection at the type level.** `PluginManager.inject_into_column_config_type_union` ORs plugin config classes into `ColumnConfigT` so Pydantic validation and static typing stay aligned with installed plugins.
- **Lazy imports via `__getattr__`.** `data_designer.config.__init__` maps public names to `(module_path, attribute_name)` and loads on first access, keeping `import data_designer.config` fast.

## Cross-References

- [System Architecture](overview.md) — package relationships and data flow
- [Engine Layer](engine.md) — how configs are compiled and executed
- [Plugins](plugins.md) — entry-point discovery and union injection
- [Sampling](sampling.md) — sampler parameter types and constraints
