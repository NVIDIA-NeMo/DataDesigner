# Plan: Processor Plugins

Created: 2026-02-03
Updated: 2026-02-19
Status: Complete

## Goal

Enable third-party processor plugins via the existing plugin discovery mechanism, and create a demo plugin package.

## Context

The callback-based processor design is already on main:
- `ProcessorConfig` base class in `processors.py`
- `Processor` base class with `process_before_batch()` and `process_after_generation()` callbacks
- `ProcessorRunner` handles all stages in the builder
- Preview runs `process_after_generation` via `run_after_generation_on_df()`

The plugin system exists for `COLUMN_GENERATOR` and `SEED_READER` types. The `_types` module pattern
(e.g., `column_types.py`, `seed_source_types.py`) separates base classes from type unions with plugin injection.

`ProcessorConfigT` is currently defined inline in `processors.py` without plugin injection.

## Success Criteria

- [x] `PluginType.PROCESSOR` enables external processor plugins
- [x] ProcessorRegistry loads plugins from entry points
- [x] Demo plugin package demonstrates both preprocessing and postprocessing
- [x] Existing `POST_BATCH` behavior unchanged

## Implementation Steps

### Step 1: Add Processor Plugin Support

Enable third-party processor plugins through the existing plugin system.

- [x] Add `PluginType.PROCESSOR` to the plugin types enum
- [x] Update `discriminator_field` property to return `"processor_type"` for processors
- [x] Update `ProcessorRegistry` to discover and load processor plugins
  - Follow the pattern used for column generator plugins
  - Use string keys for plugin processors (not enum values)

- [x] Inject plugin processor configs into the `ProcessorConfigT` type union
  - Follow the existing `_types` pattern used for columns and seed sources

**Follow the `_types` Module Pattern**:

- Keep `processors.py` with base classes and concrete configs
- Create `processor_types.py` for `ProcessorConfigT` with plugin injection
- Plugin configs import from `processors.py` (no circular dependency)

**Threading Note**:

The `PluginRegistry` uses `Lock`. If plugin discovery triggers nested imports that re-enter
the registry (e.g., a plugin imports `data_designer.config.config_builder`), this will deadlock.
Use `RLock` instead of `Lock` to allow reentrant acquisition.

Import chain that triggers this:
```
plugin.py → data_designer.config.config_builder
          → data_designer.config.column_types (calls PluginManager())
          → data_designer.config.processor_types (calls PluginManager())
          → data_designer.config.seed_source_types (calls PluginManager())
```

### Step 2: Create Demo Plugin Package

Create a separate package demonstrating both processor types.

- [x] Create package structure under `demo/data_designer_demo_processors/`
- [x] Implement `RegexFilterProcessor` (runs at `process_before_batch`)
  - Config: column, pattern, invert flag
  - Filters rows based on regex matching
- [x] Implement `SemanticDedupProcessor` (runs at `process_after_generation`)
  - Config: column, similarity_threshold, model_name
  - Uses embeddings to find and remove similar rows
  - Use sentence-transformers with a small model like `all-MiniLM-L6-v2`

- [x] Configure entry points in `pyproject.toml` under `data_designer.plugins`
- [x] Add unit tests for each processor
- [x] Add README with installation and usage examples

**Logging Suppression** (for sentence-transformers):

- Use `transformers.utils.logging.set_verbosity_error()` to suppress info/warning messages
- Use `transformers.utils.logging.disable_progress_bar()` to suppress progress bars
- Pass `show_progress_bar=False` to `model.encode()` for batch encoding

### Step 3: Demo Notebook

Create a simple, short demo that tests all features end-to-end.

- [x] Use `#%%` cell markers for IDE compatibility
- [x] Keep the demo minimal - just enough to verify the feature works
- [x] Include sample seed data with rows to filter (process_before_batch test)
- [x] Add an LLM column to generate content, use the `openai-text` model
- [x] Configure both process_before_batch and process_after_generation processors
- [x] **Run the demo and fix any issues** - don't just write it, execute it
- [x] Verify the output shows filtering and deduplication working

**Important**: The demo must actually run successfully. Test it before considering this step complete.

**API Notes**: Check the docs for correct Data Designer API usage.

### Step 4: Documentation

Update existing documentation to cover new capabilities.

- [x] Update processor concepts doc with plugin processor info
- [x] Update plugins overview to mention processor plugins
- [x] Include example entry point configuration

## Testing Strategy

- Write tests alongside implementation, not as a separate step
- Use mocks for external dependencies (seed readers, artifact storage)
- For plugin registry tests, create actual mock classes (not Mock objects) to satisfy type validation

## Risks & Considerations

- **Memory usage**: `process_after_generation` holds full dataset in memory
- **Model download**: Embedding models download on first use; perform pre-download on uv install
