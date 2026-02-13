# Agent Context CLI — Implementation Review

## Summary

This review covers the implementation of `data-designer agent-context`, a new CLI command group that exposes DataDesigner's full configuration API surface as agent-friendly introspection commands. The feature was built by porting and improving existing standalone skill scripts (`skill/data-designer/scripts/get_*.py`) into the library itself, expanding coverage from 4 config domains to the full API, and separating data extraction from presentation to support multiple output formats.

**Key stats:**
- 10 new CLI subcommands
- 8 new source files + 1 modified source file
- 6 new test files (113 tests)
- 2 output formats: plain text (YAML-style) and JSON

---

## Architecture

```
data-designer agent-context
├── columns [TYPE]         # Column types & fields
├── samplers [TYPE]        # Sampler types & params
├── validators [TYPE]      # Validator types & params
├── processors [TYPE]      # Processor types & configs
├── models                 # ModelConfig, inference params, distributions
├── builder                # DataDesignerConfigBuilder method signatures
├── constraints            # ScalarInequality, ColumnInequality, operators
├── seeds                  # SeedConfig, SeedSource types, SamplingStrategy
├── mcp                    # MCPProvider, LocalStdioMCPProvider, ToolConfig
└── overview               # Compact cheatsheet: type counts + builder summary
```

The implementation follows a layered architecture with clear separation of concerns:

```
commands/agent_context.py          # Thin Typer command wrappers
        │
        ▼
controllers/agent_context_controller.py  # Orchestration: discovery → inspection → formatting → output
        │
        ▼
services/introspection/
    ├── discovery.py               # Dynamic type discovery (8 functions)
    ├── pydantic_inspector.py      # Pydantic model introspection (dataclass-based)
    ├── method_inspector.py        # Class method introspection via inspect.signature()
    └── formatters.py              # Text and JSON output formatters
```

---

## New Files

### Source Files

All paths relative to `packages/data-designer/src/data_designer/cli/`.

| File | Lines | Purpose |
|------|-------|---------|
| `services/introspection/__init__.py` | 64 | Public exports for all introspection modules |
| `services/introspection/pydantic_inspector.py` | 257 | Core Pydantic model introspection with `FieldDetail` and `ModelSchema` dataclasses |
| `services/introspection/discovery.py` | 284 | 8 discovery functions + centralized `DEFAULT_FIELD_DESCRIPTIONS` (108 entries) |
| `services/introspection/method_inspector.py` | 251 | Class method introspection via `inspect.signature()` with Google-style docstring parsing |
| `services/introspection/formatters.py` | 183 | Text (YAML-style) and JSON formatters for all data types |
| `controllers/agent_context_controller.py` | 263 | Controller orchestrating discovery, inspection, formatting, and output |
| `commands/agent_context.py` | 115 | Typer subcommand group with 10 commands |

### Modified Files

| File | Change |
|------|--------|
| `main.py` | Added `agent_context` import and `app.add_typer(...)` registration |
| `controllers/__init__.py` | Added `AgentContextController` to exports |

### Test Files

All paths relative to `packages/data-designer/tests/cli/`.

| File | Tests | Purpose |
|------|-------|---------|
| `services/introspection/test_pydantic_inspector.py` | 38 | Unit tests for type introspection (field extraction, enum detection, nested models, cycles, depth limits) |
| `services/introspection/test_discovery.py` | 17 | Tests all 8 discovery functions find expected types |
| `services/introspection/test_method_inspector.py` | 11 | Tests docstring parsing, method signature extraction, public/private filtering |
| `services/introspection/test_formatters.py` | 19 | Tests all text and JSON formatters (schemas, methods, type lists, overview) |
| `controllers/test_agent_context_controller.py` | 15 | Controller orchestration tests using `capsys` |
| `commands/test_agent_context_command.py` | 13 | End-to-end CLI integration tests via `typer.testing.CliRunner` |
| **Total** | **113** | |

---

## Key Design Decisions

### 1. Dataclass-based structured data (not raw tuples)

The existing skill scripts used raw tuples for field information. The new implementation uses typed dataclasses:

```python
@dataclass
class FieldDetail:
    name: str
    type_str: str
    description: str
    enum_values: list[str] | None = None
    nested_schema: ModelSchema | None = None

@dataclass
class ModelSchema:
    class_name: str
    description: str
    type_key: str | None = None
    type_value: str | None = None
    fields: list[FieldDetail] = field(default_factory=list)
```

This enables clean separation between introspection and formatting, and makes JSON output trivial.

### 2. Plain text output (no Rich/ANSI)

All output uses `typer.echo()` producing plain text. Agents parse plain text more reliably than colored/ANSI output. The YAML-style text format is backward-compatible with the existing skill script output.

### 3. Dynamic discovery for extensibility

Column configs, sampler types, validator types, and processor configs are discovered dynamically by iterating `dir(data_designer.config)` and matching class name patterns. This means new types added to the config package are automatically picked up without code changes.

### 4. Cycle and depth protection

Nested model expansion uses a `seen` set (by class name) and `max_depth` parameter (default 3) to prevent infinite recursion from self-referential or deeply nested models.

### 5. Centralized field descriptions

All 108 default field descriptions are in a single `DEFAULT_FIELD_DESCRIPTIONS` dict in `discovery.py`, replacing 4 separate per-script copies.

---

## Expected Behavior

### Common Flags

For type-based commands (columns, samplers, validators, processors):
- **Positional `TYPE`**: Show details for a specific type (e.g., `llm-text`, `category`)
- **`TYPE` = `all`**: Show details for all types in the category
- **No `TYPE` (no `--list`)**: Show summary table of available types
- **`--list` / `-l`**: Show summary table of available types
- **`--format json` / `-f json`**: JSON output instead of text

For other commands (models, builder, constraints, seeds, mcp, overview):
- **`--format json` / `-f json`**: JSON output instead of text

### Command-by-Command Behavior

#### `data-designer agent-context columns`

Shows column configuration types discovered from `data_designer.config`.

```bash
# List all column types
$ data-designer agent-context columns --list
column_type    config_class
-----------    -------------------------
custom         CustomColumnConfig
embedding      EmbeddingColumnConfig
expression     ExpressionColumnConfig
llm-code       LLMCodeColumnConfig
llm-judge      LLMJudgeColumnConfig
llm-structured LLMStructuredColumnConfig
llm-text       LLMTextColumnConfig
sampler        SamplerColumnConfig
seed-dataset   SeedDatasetColumnConfig
validation     ValidationColumnConfig

# Show details for a specific type
$ data-designer agent-context columns llm-text
LLMTextColumnConfig:
  column_type: llm-text
  description: Configuration for LLM-based text generation columns.
  fields:
    name:
      type: str
      description: Unique column name in the generated dataset
    prompt:
      type: str
      description: Jinja2 template for the LLM prompt...
    ...

# JSON format
$ data-designer agent-context columns llm-text --format json
{
  "class_name": "LLMTextColumnConfig",
  "description": "Configuration for LLM-based text generation columns.",
  "column_type": "llm-text",
  "fields": [
    {"name": "name", "type": "str", "description": "..."},
    ...
  ]
}

# Unknown type exits with error
$ data-designer agent-context columns nonexistent
Error: Unknown column_type 'nonexistent'
Available types: custom, embedding, expression, ...
```

#### `data-designer agent-context samplers`

Shows sampler types discovered from `SamplerType` enum and their params classes. Type lookups are case-insensitive, and the type value is displayed in uppercase (e.g., `CATEGORY`).

```bash
$ data-designer agent-context samplers category
CategorySamplerParams:
  sampler_type: CATEGORY
  description: ...
  fields:
    values:
      type: list[str]
      description: List of categorical values to sample from
    ...
```

#### `data-designer agent-context validators`

Shows validator types (CODE, REMOTE, LOCAL_CALLABLE) and their params classes. Same pattern as samplers.

#### `data-designer agent-context processors`

Shows processor types (drop_columns, templated_columns) and their config classes.

#### `data-designer agent-context models`

Shows all model-related types: `ModelConfig`, `ChatCompletionInferenceParams`, `EmbeddingInferenceParams`, `ImageInferenceParams`, `ImageContext`, `UniformDistribution`, `ManualDistribution`.

```bash
$ data-designer agent-context models
# Data Designer Model Configuration Reference
# 7 types

ChatCompletionInferenceParams:
  description: ...
  fields:
    temperature:
      type: float | UniformDistribution | ManualDistribution | None
      ...
...
```

#### `data-designer agent-context builder`

Shows `DataDesignerConfigBuilder` method signatures and documentation, extracted via `inspect.signature()` and Google-style docstring parsing.

```bash
$ data-designer agent-context builder
DataDesignerConfigBuilder Methods:

  add_column(column: ColumnConfig) -> Self
    Add a column configuration to the builder.
    Parameters:
      column: ColumnConfig — The column configuration to add.
  ...
```

#### `data-designer agent-context constraints`

Shows constraint types: `ScalarInequalityConstraint`, `ColumnInequalityConstraint`, `InequalityOperator`.

#### `data-designer agent-context seeds`

Shows seed dataset types: `SeedConfig`, `SamplingStrategy`, `LocalFileSeedSource`, `HuggingFaceSeedSource`, `DataFrameSeedSource`, `IndexRange`, `PartitionBlock`.

#### `data-designer agent-context mcp`

Shows MCP types: `MCPProvider`, `LocalStdioMCPProvider`, `ToolConfig`.

#### `data-designer agent-context overview`

Compact API cheatsheet with type counts, builder method summaries, and quick-start commands.

```bash
$ data-designer agent-context overview
Data Designer API Overview
==========================

Type Counts:
  Column types:     10
  Sampler types:    12
  Validator types:   3
  Processor types:   2
  Model configs:     7
  Constraint types:  3
  Seed types:        7
  MCP types:         3

Builder Methods (DataDesignerConfigBuilder):
  add_column(...)         — Add a column configuration to the builder.
  add_constraint(...)     — Add a constraint to the builder.
  ...

Quick Start Commands:
  data-designer agent-context columns --list
  data-designer agent-context columns all
  data-designer agent-context columns llm-text
  data-designer agent-context samplers category
  data-designer agent-context builder
```

---

## Improvements Over Skill Scripts

| Aspect | Skill Scripts | Agent Context CLI |
|--------|--------------|-------------------|
| **Location** | External (`skill/data-designer/scripts/`) | Library (`data_designer.cli`) |
| **Data structure** | Raw tuples, print directly | Dataclasses (`FieldDetail`, `ModelSchema`, `MethodInfo`, `ParamInfo`) |
| **Output formats** | Text only | Text + JSON (`--format json`) |
| **API coverage** | 4 domains (columns, samplers, validators, processors) | 9 domains (+models, builder, constraints, seeds, MCP, overview) |
| **Field descriptions** | 4 separate dicts | 1 centralized dict (108 entries) |
| **Builder introspection** | None | Full method signatures + docstring parsing |
| **Error handling** | Varies | Consistent: error message + exit code 1 |
| **Testability** | Script-level test | 113 unit + integration tests at every layer |

---

## Test Coverage Summary

### Unit Tests (85 tests)

**`test_pydantic_inspector.py` (38 tests):**
- `_is_basemodel_subclass`: 5 tests (subclass, BaseModel itself, str, enum, non-type)
- `_is_enum_subclass`: 4 tests (subclass, Enum itself, str, non-type)
- `_extract_enum_class`: 5 tests (direct, optional, annotated, non-enum, None)
- `extract_nested_basemodel`: 10 tests (direct, list, optional, optional-list, dict, annotated, discriminated union, primitive, None, BaseModel itself)
- `format_type`: 3 tests (str, int, optional)
- `get_brief_description`: 2 tests (with/without docstring)
- `get_field_info`: 4 tests (returns FieldDetails, default descriptions, enum values, non-enum)
- `build_model_schema`: 5 tests (basic structure, type key/value, nested expansion, cycle protection, depth limiting)

**`test_discovery.py` (17 tests):**
- 2 tests per discovery function (returns dict + contains expected keys) for all 8 functions
- 1 extra test for `discover_column_configs` (values are classes with model_fields)

**`test_method_inspector.py` (11 tests):**
- `_parse_google_docstring_args`: 4 tests (basic, empty, no args section, multiline)
- `inspect_class_methods`: 7 tests (public only, returns MethodInfo, signature content, description, parameters, include private, init included)

**`test_formatters.py` (19 tests):**
- `format_model_schema_text`: 4 tests (basic, type key, nested, enum values)
- `format_model_schema_json`: 3 tests (basic, type key, nested)
- `format_method_info_text`: 3 tests (basic, with class name, without)
- `format_method_info_json`: 2 tests (basic, multiple methods)
- `format_type_list_text`: 3 tests (basic, alignment, empty)
- `format_overview_text`: 4 tests (header, type counts, builder methods, quick start)

### Controller Tests (15 tests)

**`test_agent_context_controller.py`:**
- `show_columns`: 6 tests (list mode, specific type, all, nonexistent exits, JSON format, list JSON)
- `show_overview`: 2 tests (text, JSON)
- `show_samplers`: 2 tests (list, specific)
- `show_models`: 1 test
- `show_builder`: 1 test
- `show_constraints`: 1 test
- `show_seeds`: 1 test
- `show_mcp`: 1 test

### Integration Tests (13 tests)

**`test_agent_context_command.py`:**
- Via `typer.testing.CliRunner` against the real `app`:
  - `agent-context --help`
  - `columns --list`, `columns llm-text`, `columns llm-text --format json`, `columns nonexistent`
  - `samplers category`, `samplers --list`
  - `overview`, `builder`, `models`, `constraints`, `seeds`, `mcp`

---

## Verification Checklist

- [x] `make check-all` passes (ruff format + lint)
- [x] All 113 new tests pass
- [x] All 670 total project tests pass (113 new + 557 existing)
- [x] SPDX license headers on all files (2025-2026)
- [x] Type annotations on all functions
- [x] Absolute imports only (no relative imports)
- [x] No in-function imports (except `data_designer.config` in discovery functions, which is intentional to avoid circular imports at module load time)
- [x] Plain text output (no Rich/ANSI) for agent compatibility
- [x] JSON output is valid `json.loads()`-parseable
- [x] Error handling: unknown types produce clear error message + exit code 1
- [x] Backward compatibility: YAML-style text format matches existing skill script output
