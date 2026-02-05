# Plan: Processor Plugins with Global Stages

Created: 2026-02-03
Status: Iterating and Refining

## Goal

Extend the processor system to support global preprocessing (before generation) and postprocessing (after generation) stages, and enable third-party processor plugins via the existing plugin discovery mechanism.

## Success Criteria

- [x] `PRE_GENERATION` stage runs once on full seed data before batching/generation
- [x] `POST_GENERATION` stage runs once on final dataset after all batches complete
- [x] `PluginType.PROCESSOR` enables external processor plugins
- [x] ProcessorRegistry loads plugins from entry points
- [x] Demo plugin package demonstrates both preprocessing and postprocessing
- [x] Existing `POST_BATCH` behavior unchanged

## Implementation Steps

### Step 1: Extend BuildStage Support

Update processor configuration to accept new stages.

- [x] Add `PRE_GENERATION` and `POST_GENERATION` to `SUPPORTED_STAGES` in processors.py
- [x] Add unit tests verifying `ProcessorConfig` accepts the new stage values

**Suggestion**: Check if `BuildStage` enum already has these values defined elsewhere before adding.

### Step 2: Update Dataset Builder for Global Stages

Implement the actual execution of processors at the new stages.

- [x] Add `_run_pre_generation_processors()` method
  - Load full seed dataset before batch loop
  - Apply PRE_GENERATION processors sequentially
  - Replace the seed reader with an in-memory version containing processed data
  - **Suggestion**: Look for existing in-memory seed reader implementations (e.g., `DataFrameSeedReader`)

- [x] Add `_run_post_generation_processors()` method
  - Load the final combined dataset after all batches complete
  - Apply POST_GENERATION processors sequentially
  - Rewrite the final dataset with processed results
  - **Suggestion**: Check how existing artifact storage handles dataset loading/writing

- [x] Integrate calls into the `build()` method at appropriate points
- [x] Add integration tests for both flows

### Step 3: Add Processor Plugin Support

Enable third-party processor plugins through the existing plugin system.

- [x] Add `PluginType.PROCESSOR` to the plugin types enum
- [x] Update `discriminator_field` property to return `"processor_type"` for processors
- [x] Update `ProcessorRegistry` to discover and load processor plugins
  - **Suggestion**: Follow the pattern used for column generator plugins
  - Use string keys for plugin processors (not enum values)

- [x] Inject plugin processor configs into the `ProcessorConfigT` type union
  - Follow the existing `_types` pattern used for columns and seed sources

**Follow the `_types` Module Pattern**:

The codebase separates base classes from type unions with plugin injection:
- `column_configs.py` (base) → `column_types.py` (union + injection)
- `seed_source.py` (base) → `seed_source_types.py` (union + injection)

Do the same for processors:
- [x] Keep `processors.py` with base classes and concrete configs
- [x] Create `processor_types.py` for `ProcessorConfigT` with plugin injection
- [x] Plugin configs import from `processors.py` (no circular dependency)

**Threading Note**:

If you encounter deadlocks during plugin discovery with nested imports, the `PluginRegistry` may need a reentrant lock (`RLock`) instead of `Lock`.

### Step 4: Create Demo Plugin Package

Create a separate package demonstrating both processor types.

- [x] Create package structure under `demo/data_designer_demo_processors/`
- [x] Implement `RegexFilterProcessor` (PRE_GENERATION)
  - Config: column, pattern, invert flag
  - Filters rows based on regex matching
- [x] Implement `SemanticDedupProcessor` (POST_GENERATION)
  - Config: column, similarity_threshold, model_name
  - Uses embeddings to find and remove similar rows
  - **Suggestion**: Use sentence-transformers with a small model like `all-MiniLM-L6-v2`

- [x] Configure entry points in `pyproject.toml` under `data_designer.plugins`
- [x] Add unit tests for each processor
- [x] Add README with installation and usage examples

**Logging Suppression** (for sentence-transformers):

Sentence-transformers emits progress bars and warnings when loading models. Suppress them:

- Use `transformers.utils.logging.set_verbosity_error()` to suppress info/warning messages
- Use `transformers.utils.logging.disable_progress_bar()` to suppress progress bars
- Pass `show_progress_bar=False` to `model.encode()` for batch encoding

### Step 5: Demo Notebook

Create a simple, short demo that tests all features end-to-end.

- [x] Use `#%%` cell markers for IDE compatibility
- [x] Keep the demo minimal - just enough to verify the feature works
- [x] Include sample seed data with rows to filter (PRE_GENERATION test)
- [x] Add an LLM column to generate content, use the `openai-text` model
- [x] Configure both PRE_GENERATION and POST_GENERATION processors
- [x] **Run the demo and fix any issues** - don't just write it, execute it
- [x] Verify the output shows filtering and deduplication working

**Important**: The demo must actually run successfully. Test it before considering this step complete.

**API Notes**: Check the docs for correct Data Designer API usage.

### Step 6: Documentation

Update existing documentation to cover new capabilities.

- [x] Update processor concepts doc with new stages table
- [x] Update plugins overview to mention processor plugins
- [x] Include example entry point configuration

## Testing Strategy

- Write tests alongside implementation, not as a separate step
- Use mocks for external dependencies (seed readers, artifact storage)
- For plugin registry tests, create actual mock classes (not Mock objects) to satisfy type validation

## Risks & Considerations

- **Memory usage**: POST_GENERATION holds full dataset in memory
- **Seed data mutation**: PRE_GENERATION modifies seed data before batching
- **Model download**: Embedding models download on first use; perform pre-download on uv install

## Files Modified

Core:
- `packages/data-designer-config/src/data_designer/config/processor_types.py` (new)
- `packages/data-designer-config/src/data_designer/config/processors.py`
- `packages/data-designer-config/src/data_designer/config/data_designer_config.py`
- `packages/data-designer-config/src/data_designer/config/config_builder.py`
- `packages/data-designer-config/src/data_designer/plugin_manager.py`
- `packages/data-designer-config/src/data_designer/plugins/plugin.py`
- `packages/data-designer-config/src/data_designer/plugins/registry.py`
- `packages/data-designer-engine/src/data_designer/engine/dataset_builders/column_wise_builder.py`
- `packages/data-designer-engine/src/data_designer/engine/processing/processors/registry.py`
- `packages/data-designer-engine/src/data_designer/engine/validation.py`

Demo:
- `demo/data_designer_demo_processors/` (new package)

Docs:
- `docs/concepts/processors.md`
- `docs/plugins/overview.md`

---

## Iterating and Refining

### Issue 1: Preview does not apply PRE_GENERATION / POST_GENERATION processors

**Problem**: `build_preview()` and `process_preview()` do not call the new global-stage processors. This means users previewing their data pipeline won't see the effects of filtering or deduplication until they run a full build.

**Investigation**:
- `build()` calls `_run_pre_generation_processors()` before generation and `_run_post_generation_processors()` after
- `build_preview()` skips both
- `process_preview()` only applies `POST_BATCH` processors

**Fix**:
- [x] Add `_run_pre_generation_processors()` call to `build_preview()` before `_initialize_generators()`
- [x] Update `process_preview()` to also run `POST_GENERATION` processors after `POST_BATCH`
- [x] Add tests for preview with global-stage processors

### Issue 2: Is RLock necessary in PluginRegistry?

**Question**: The plan suggested changing `Lock` to `RLock` in `PluginRegistry`. Is this actually needed?

**Investigation**:

The `PluginRegistry` singleton uses a lock in three places:
- `__new__`: double-checked locking for singleton creation
- `__init__`: protects `_discover()` call
- `reset()`: resets singleton state

The potential deadlock scenario:
1. `PluginRegistry.__init__` acquires lock
2. `_discover()` calls `ep.load()` to load a plugin module
3. The plugin imports `data_designer.config.config_builder` (or any module using the config API)
4. That imports `column_types.py`, `processor_types.py`, `seed_source_types.py`
5. Each of those calls `PluginManager()` → `PluginRegistry()` at module level
6. `PluginRegistry().__init__` tries to acquire lock again (same thread)
7. With `Lock`: deadlock (same thread blocked). With `RLock`: succeeds.

Import chain that triggers this:
```
plugin.py → data_designer.config.config_builder
          → data_designer.config.column_types (calls PluginManager())
          → data_designer.config.processor_types (calls PluginManager())
          → data_designer.config.seed_source_types (calls PluginManager())
```

**Conclusion**: YES, `RLock` is necessary. Any third-party plugin that imports from the `data_designer.config` public API (e.g., `DataDesignerConfigBuilder`, `DataDesignerConfig`) would trigger this re-entry. Using a regular `Lock` would cause a deadlock.

### Iteration 3: Callback-based Processor Design

**Date**: 2026-02-04

**Issue**: The `build_stage` config field determines when a processor runs, but processors are inherently tied to specific stages. A `SemanticDedupProcessor` running at PRE_GENERATION doesn't make sense - it's meant to deduplicate final outputs. This creates a semantic mismatch where config controls something that's really a property of the processor class.

**Root Cause**: Original design treated stage as configuration rather than as part of the processor's interface. This:
- Polluted config with fields that aren't user-configurable in practice
- Made single-stage limitation implicit (processors can only run at one stage)
- Created inconsistent method signatures (`current_batch_number` only meaningful for POST_BATCH)

**Resolution**: Changed from stage-based config to callback-based design:

1. **Removed `build_stage` from `ProcessorConfig`** - Processors no longer declare when they run via config

2. **Added callback methods to `Processor` base class**:
   - `preprocess(data)` - Called at PRE_GENERATION on seed data
   - `process_after_batch(data, *, batch_number)` - Called at POST_BATCH for each batch
   - `postprocess(data)` - Called at POST_GENERATION on final dataset

3. **Changed `_processors` from `dict[BuildStage, list]` to `list[Processor]`** - All processors in flat list, callbacks called on all

4. **Updated built-in processors**:
   - `DropColumnsProcessor` → overrides `process_after_batch()`
   - `SchemaTransformProcessor` → overrides `process_after_batch()`

5. **Updated demo processors**:
   - `RegexFilterProcessor` → overrides `preprocess()`
   - `SemanticDedupProcessor` → overrides `postprocess()`

**Benefits**:
- Self-documenting: Looking at a processor class shows what stages it handles
- Multi-stage capable: One processor can implement multiple callbacks
- Clean signatures: Each callback has only relevant parameters
- Simpler config: No `build_stage` field to confuse users

**Verification**:
- All 1360 tests pass
- Config tests updated to remove `build_stage` parameter
- Builder tests updated for new callback pattern
