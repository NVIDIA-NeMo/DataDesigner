---
date: 2026-08-10
authors:
  - andreatnvidia
issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/851
epic: https://github.com/NVIDIA-NeMo/DataDesigner/issues/850
status: proposal
---

# Public Data Designer invocation contract

## Summary

An optional execution package can configure, validate, and invoke Data Designer without importing
`data_designer.engine`. The supported boundary consists of serialized builder input, public config models,
`DataDesigner`, `DatasetCreationResults`, and public errors.

Plugin-independent envelope validation may run before optional plugin packages are installed. Full builder validation
must run in a fresh client process after those packages are installed and available on `sys.path`.

## Public imports

The execution package may import these public symbols:

```python
from data_designer.config import (
    DataDesignerConfigBuilder,
    InvalidConfigError,
    InvalidFileFormatError,
    InvalidFilePathError,
    LocalStdioMCPProvider,
    MCPProvider,
    ModelConfig,
    ModelProvider,
    PartitionBlock,
    ResumeMode,
    RunConfig,
    SamplingStrategy,
)
from data_designer.interface import (
    ArtifactStorageError,
    DataDesigner,
    DataDesignerEarlyShutdownError,
    DataDesignerGenerationError,
    DataDesignerProfilingError,
    DatasetCreationResults,
)
from data_designer.plugins import Plugin
```

The package must not depend on:

- Any `data_designer.engine` import.
- `data_designer.config.config_builder.BuilderConfig`.
- `data_designer.config.mcp.MCPProviderT`.
- `DatasetCreationResults.artifact_storage` or `task_traces`.
- The plugin registry implementation.
- Typed errors documented by `DataDesigner.check_models()` outside the public interface error module.

`DataDesigner` may use engine modules internally. The restriction applies to imports and types crossing the package
boundary.

## Serialized builder boundary

Authored input contains exactly one of:

- A path to a complete JSON or YAML builder config.
- The same builder config as an inline mapping.

The payload is the format written by `DataDesignerConfigBuilder.write_config()`. The public loader is:

```python
builder = DataDesignerConfigBuilder.from_config(builder_payload_or_path)
```

`BuilderConfig` is an implementation type used by the serializer. It is not part of the cross-package API.
Invocation controls such as record count, dataset name, resume, runtime settings, model endpoints, and output format
remain outside the builder payload.

## Validation ownership

### Submission environment

The submission environment validates only information that does not require plugin-specific config classes:

- Outer execution schema and version.
- Builder source location, format, digest, and top-level mapping shape.
- Unique aliases declared in the raw `data_designer.model_configs` list.
- Deployment coverage for those declared aliases.
- `RunConfig`, output, image, dependency, and resource fields owned by the execution package.

It must not instantiate the complete builder when referenced plugins are unavailable. Plugin columns may declare
additional model references, so referenced-alias validation is deferred with full builder validation.

### Client environment

The package overlay must be installed before the client process imports Data Designer config unions. Plugin discovery
is process-global and happens on first use, so installing a package after config import is not a supported refresh
path.

Client preflight runs in a fresh process and performs these steps:

1. Enumerate the expected `data_designer.plugins` entry points with `importlib.metadata`.
2. Load each entry point and verify that it returns a public `Plugin` object.
3. Resolve `Plugin.config_cls` and `Plugin.impl_cls` so import failures stop preflight.
4. Load the builder with `DataDesignerConfigBuilder.from_config()`.
5. Materialize model bindings, per-model concurrency, seed input bindings, and MCP providers.
6. Verify referenced model aliases through each column config's public `get_model_aliases()` method and each profiler
   config's public `model_alias` field when present.
7. Construct `DataDesigner`, apply `RunConfig`, and call `DataDesigner.validate(builder)`.

The explicit entry-point load is required because normal plugin discovery logs and skips a failing entry point.
`DataDesigner.validate()` validates config structure and seed-dependent compilation without contacting model endpoints.
It does not verify that every referenced column or profiler alias has a matching model config, so client preflight owns
that check. Endpoint readiness belongs after services are available; generation performs its normal readiness check.

## Model binding

`ModelConfig.alias` is the workload identity. The original `model` string is not an alias and must not be used to
match deployments.

For every declared alias, the client creates a stable `ModelProvider` for the resolved logical endpoint and replaces
the matching `ModelConfig` while preserving unrelated inference parameters:

```python
originals = list(builder.model_configs)
for model_config in originals:
    builder.delete_model_config(model_config.alias)

for model_config in originals:
    binding = bindings_by_alias[model_config.alias]
    inference_parameters = model_config.inference_parameters.model_copy(
        update={"max_parallel_requests": binding.max_parallel_requests}
    )
    builder.add_model_config(
        model_config.model_copy(
            update={
                "model": binding.served_model,
                "provider": binding.provider_name,
                "inference_parameters": inference_parameters,
            }
        )
    )
```

Deleting all originals before adding replacements preserves alias uniqueness and original ordering. Full preflight
rejects missing or duplicate declared aliases, missing referenced aliases, missing providers, and deployment aliases
that do not match `ModelConfig.alias`.

Per-model concurrency is `ModelConfig.inference_parameters.max_parallel_requests`. It is not a `RunConfig` field.

## Runtime configuration

The execution package merges its compatibility defaults with the raw authored `run_config` mapping, then validates
the effective mapping once:

```python
effective_run_config = RunConfig.model_validate(compatibility_defaults | authored_run_config)
data_designer.set_run_config(effective_run_config)
```

Merging must occur before constructing the authored `RunConfig`. Model validators normalize related fields, so a
validated model's field-set metadata is not the authoritative record of which keys the user wrote.

The effective default for every public field is:

| Field | Data Designer default | Execution-package default |
| --- | ---: | ---: |
| `disable_early_shutdown` | `False` | `True` when no early-shutdown control is authored |
| `shutdown_error_rate` | `0.5` | `1.0` after disabled-shutdown normalization |
| `shutdown_error_window` | `10` | `10` |
| `buffer_size` | `1000` | `16384` |
| `max_concurrent_row_groups` | `3` | `3` |
| `max_in_flight_tasks` | `1024` | `1024` |
| `non_inference_max_parallel_workers` | `4` | `4` |
| `max_conversation_restarts` | `5` | `0` |
| `max_conversation_correction_steps` | `0` | `0` |
| `async_trace` | `False` | `False` |
| `write_scheduler_events` | `False` | `False` |
| `display_tui` | `False` | `False` |
| `progress_interval` | `5.0` | `5.0` |
| `otel_metrics_port` | `9464` | `None` |
| `preserve_dropped_columns` | `True` | `True` |
| `jinja_rendering_engine` | `secure` | `secure` |
| `request_admission` | `None` | `None` |

Every authored key overrides the corresponding execution-package default. Early-shutdown controls form one related
group: if the user authors `shutdown_error_rate` or `shutdown_error_window` without `disable_early_shutdown`, the
package retains Data Designer's enabled-shutdown default instead of injecting `disable_early_shutdown=True` and
discarding the authored threshold.

`non_inference_max_parallel_workers` is a public field but is not currently consumed by generation. The execution
package must not claim CPU-derived worker control until Data Designer implements that behavior.

## Invocation

The client materializes provider and MCP secrets immediately before constructing public config objects. Secret
values do not enter authored or persisted config.

```python
data_designer = DataDesigner(
    artifact_path=dataset_workspace,
    model_providers=model_providers,
    managed_assets_path=managed_assets_path,
    mcp_providers=mcp_providers,
    auto_configure_logging=False,
)
data_designer.set_run_config(effective_run_config)
data_designer.validate(builder)

results = data_designer.create(
    builder,
    num_records=requested_num_records,
    dataset_name=dataset_name,
    resume=resume_mode,
)
```

`DataDesigner.acreate()` is the equivalent non-blocking entry point. It delegates generation to a worker thread and
returns the same `DatasetCreationResults` type.

Remote MCP connections use `MCPProvider`; local subprocess connections use `LocalStdioMCPProvider`. The execution
package owns any secret-reference schema and passes only resolved strings to Data Designer.

## Results and errors

A normal return means generation and profiling completed without a public exception. It does not guarantee that the
dataset contains exactly the requested number of records.

`DatasetCreationResults` exposes:

| Field or method | Meaning |
| --- | --- |
| `dataset_path` | Resolved dataset directory, including collision or resume resolution. |
| `requested_num_records` | Target passed to `create()`, or the persisted target for a reconstructed workflow result. |
| `actual_num_records` | Current total rows in the final dataset, including rows from an earlier resumed invocation. |
| `is_partial` | `True` when actual records are fewer than requested records, or `None` when the target is unavailable. |
| `early_shutdown` | Whether the current invocation stopped through the early-shutdown gate, or `None` when no generation invocation produced the result object. |
| `requested_resume_mode` | Resume mode passed to `create()`, or `None` when no generation invocation produced the result object. |
| `effective_resume_mode` | `always` when the invocation resumed, `never` when it started fresh, or `None` when no generation invocation produced the result object. |
| `count_records()` | Metadata-only row count, equivalent to `actual_num_records`. |
| `export(path, format=...)` | Stream the result to one JSONL, CSV, or Parquet file. |

The caller derives its own exact-count policy from `actual_num_records == requested_num_records`. A partial result may
be caused by dropped rows or early shutdown; `early_shutdown` distinguishes those cases.

Public failure behavior is:

| Condition | Public behavior |
| --- | --- |
| Missing or unreadable local builder file | `InvalidFilePathError` from `data_designer.config`. |
| Malformed local or inline builder data | `InvalidFileFormatError` from `data_designer.config`. |
| Invalid serialized builder shape or remote source | `pydantic.ValidationError` or `ValueError` from `DataDesignerConfigBuilder.from_config()`. |
| Invalid compiled config | `InvalidConfigError` from `data_designer.config`. |
| Generation failure | `DataDesignerGenerationError`. |
| Profiling failure after generation | `DataDesignerProfilingError`. No successful result is returned. |
| Early shutdown with zero records | `DataDesignerEarlyShutdownError`, a `DataDesignerGenerationError` subclass. |
| Early shutdown with some records | A partial `DatasetCreationResults` with `early_shutdown=True`. |
| Invalid export format or incompatible Parquet schemas | `InvalidFileFormatError` from `data_designer.config`. |
| Missing or unreadable dataset or processor artifacts | `ArtifactStorageError` from `data_designer.interface`. |

The caller must not inspect engine storage or task-trace types to classify an outcome.

Failure exceptions do not carry a `DatasetCreationResults` object. The resolved dataset path, actual record count,
early-shutdown state, and effective resume mode may therefore be unavailable after a failed invocation. A semantic
failure record owned by an embedding package must make those facts optional rather than inspect engine storage.

## Output, resume, seed, and telemetry

- `artifact_path`, `dataset_name`, and `ResumeMode` define the resumable workspace.
- `dataset_path` is the resolved public location. This is important when `ResumeMode.IF_POSSIBLE` starts fresh or a
  non-resumable name collides.
- Export format is explicit. Data Designer does not publish a default export-format constant.
- Seed datasets are applied with `with_seed_dataset()` and can be partitioned with public `PartitionBlock`.
- `PartitionBlock` does not define a deterministic generation random seed. No public generation-seed API exists.
- OpenTelemetry metrics are configured with `RunConfig.otel_metrics_port`; `None` disables metrics for the invocation.
- Embedded callers should use `auto_configure_logging=False` when they own process logging.

`DatasetCreationResults` does not expose per-invocation usage or phase timing. The public OpenTelemetry endpoint
provides create duration, generated/dropped record counters, and request-duration histograms, but not token totals or
separate generation and profiling durations. Callers must treat unavailable values as optional unless a future public
result summary provides them; logs and engine usage types are not a supported substitute.

## Sharded execution limits

Public builder methods expose column, processor, and profiler configs through `get_column_configs()`,
`get_processor_configs()`, and `get_profilers()`. Data Designer does not publish capability metadata that identifies
whether a processor, profiler, media output, or plugin is safe to merge across independently generated partitions.
An embedding package must use a conservative policy and reject multi-partition execution when those semantics are
present or unknown. Ordered seed input can use `PartitionBlock`; `SamplingStrategy.SHUFFLE` has no deterministic
partition-then-shuffle contract.

## Callable and plugin limits

Entry-point plugins with serializable config models are supported after installation in the client environment.

`CustomColumnConfig.generator_function` and `LocalCallableValidatorParams.validation_function` are not portable
serialized references. Their serializers write only a function name while validation requires an in-memory callable.
Installing a package later does not resolve that string. A workload requiring these callables needs a separately
approved installed builder factory or a future qualified-callable-reference feature.

## Import baseline

The base import budget remains the existing `make perf-import CLEAN=1` measurement:

```text
import data_designer.config as dd
from data_designer.interface import DataDesigner
```

The average of one cold and four warm runs must remain below three seconds. Optional command discovery may read
distribution metadata, but must not import the optional package until its command is selected. CLI extension discovery
is tracked separately by #853.

## Contract tests

The owning packages must cover:

- Public root imports and absence of external `data_designer.engine` imports.
- Builder file and inline mapping loading after plugin installation.
- Plugin entry-point load failure before model services start.
- Model binding by alias, order preservation, per-model concurrency, profiler aliases, and plugin secondary aliases.
- Compatibility-default merging from raw authored keys.
- Complete, partial, partial early-shutdown, zero-record early-shutdown, generation-error, and profiling-error outcomes.
- Failure records that allow result-only path, count, early-shutdown, and resume facts to be unavailable.
- Resolved dataset paths for fresh, colliding, resumed, and incompatible `if_possible` invocations.
- Explicit JSONL, CSV, and Parquet exports.
- Remote and stdio MCP providers without persisted secret values.
- Conservative multi-partition rejection for unsafe or unknown processor, profiler, media, and plugin semantics.
- The existing import-performance threshold.
