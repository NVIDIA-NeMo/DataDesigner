---
date: 2026-10-17
authors:
  - namanraii
issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/996
status: proposal
---

# Plan: Order-independent drop flags for `DropColumnsProcessorConfig`

## Summary

Whether a column ends up with `drop=True` currently depends on the order of builder calls.
`DataDesignerConfigBuilder.add_processor` sets the flag as a side effect, only on columns that already exist
(`config_builder.py`, `add_processor` and `_resolve_drop_column_names`). Adding the processor before the column,
or re-adding the column afterwards (an upsert that replaces the config object), loses the flag.
`_remove_processor_by_name` also resets `drop` to `False` on columns the user declared with `drop=True` themselves.

This plan derives the effective flag from the complete configuration instead of mutating column configs
while processors are added.

## Problem

Reported in #996 and confirmed by a maintainer. Three call orders with the same intent produce different configs:

| Order | `tmp` drop flag on `build()` |
| --- | --- |
| columns, then processor | `True` |
| processor, then columns | `False` |
| columns, processor, then `add_column("tmp")` again | `False` |

With `RunConfig(preserve_dropped_columns=False)` the second order fails at the end of a run:
`DataDesignerProfilingError: Column 'tmp' not found in dataset`, because the column is gone from the dataset but still
handed to the profiler.

Two related defects in the same area:

1. Replacing a drop processor clears `drop` on columns the user flagged explicitly.
2. `DatasetBuilder._initialize_processors` compares `config.column_names` to the flagged columns literally, so a glob
   such as `"tmp_*"` never matches and a redundant `default_drop_columns_processor` is added on top of the explicit one
   (extra "Dropping columns" log line and "Cannot drop column ... not found" warnings).

## Goals

- The same builder calls in any order produce the same built config.
- Re-adding a column does not lose a flag a processor implies.
- A `drop=True` the user declared is never cleared by a processor change.
- Glob targets do not produce a duplicate default drop processor.

## Non-goals

- Changing what `DropColumnsProcessorConfig` does at generation time.
- Changing validation of drop targets (`validate_drop_columns_processor`).
- Broader cleanup of processor semantics.

## Design

Follows the direction in the maintainer's comment on #996.

1. **Builder stops mutating columns.** `add_processor` only upserts the processor. `_remove_processor_by_name` only
   removes it. The rollback logic goes away, which also fixes the explicit-flag bug.
2. **Derive flags in `build()`.** `build()` resolves the targets of every drop processor against the complete set of
   column configs (globs included) and returns column configs with `drop=True` for those. Columns that need the flag
   are copied (`model_copy`), so the builder's own configs are never changed and repeated builds cannot keep a stale
   derived flag. Declared `drop=True` is preserved because the flag is only ever set, never cleared.
3. **Seed columns stay in the compiler.** The builder does not know seed columns, so
   `_mark_processor_dropped_seed_columns` in `engine/compiler.py` keeps handling them. The engine/config layering
   is unchanged: the engine does not import from the builder and the builder does not import engine code.
4. **Glob-aware default drop processor.** `_initialize_processors` removes every column an explicit drop processor
   targets, including glob matches, using the existing `resolve_processor_dropped_columns` from
   `engine/validation.py`, before deciding whether a default drop processor is needed.
5. **Display code reads the built config.** `display_sample_record` read `column.drop` from the builder's column
   configs. Since processor-implied flags now exist only on the built config, it takes the dropped names from
   `build()` so it does not look up columns a processor removed from the record.
6. **Docs.** `fern/.../concepts/processors.mdx` said columns are marked "when this processor is added"; reword.

## Behaviour change to be aware of

`builder.get_column_config(name).drop` no longer turns `True` as a side effect of `add_processor`. It reflects only
what the user declared. `build()` carries the effective flag. `get_builder_config()` and `write_config()` export the
authored column flags, so a builder reloaded with `from_config` recomputes processor-derived drops and a later processor
replacement behaves the same as on the original builder. Directly serializing the output of `build()` has snapshot
semantics: its drop flags are the effective ones and will load as explicit declarations. Compiled run artifacts stay
effective execution snapshots, which the resume fingerprint checks compare.

## Alternatives considered

- **Resolve only in the engine compiler.** Covers the run itself, but leaves `build()` output order-dependent, which is
  what the issue reproduces, and the profiler/visualization consumers read the built config.
- **Move the resolver into `data-designer-config`** so builder and engine share one implementation. Cleaner long term,
  but a larger move; the builder keeps a small resolver of its own, as it already did.

## Testing

- `data-designer-config`: build-time flags for columns-then-processor, processor-then-columns, re-added column, glob
  targets added before columns, builder not mutated across repeated builds, explicit `drop=True` kept on processor
  replacement, and `display_sample_record` with a processor-dropped column.
- `data-designer-engine`: no `default_drop_columns_processor` when an explicit processor targets columns by glob.
- `data-designer` (interface): sampler-only `create()` with `preserve_dropped_columns=False` for both call orders;
  profiling no longer fails for processor-before-column.

## Delivery

One PR against `main`, no migration needed.
