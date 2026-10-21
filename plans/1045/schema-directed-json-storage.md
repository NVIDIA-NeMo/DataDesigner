---
date: 2026-10-19
status: draft-pr
issue: https://github.com/NVIDIA-NeMo/DataDesigner/issues/1045
related_pr: https://github.com/NVIDIA-NeMo/DataDesigner/pull/1046
---

# Plan: Schema-directed JSON storage for flexible column values

## Summary

Add explicit JSON storage declarations that preserve selected JSON values through Parquet artifacts.
Generation, validation, templates, processors, profiling, and DD result loading continue to use logical
Python values. A shared storage boundary encodes values before writing and decodes them after reading.
External Parquet readers see the documented physical representation.

For structured outputs, declare storage on a schema node with
`"x-data-designer-storage": "json"`. Compile this separately from the generation schema before model
calls begin. Persist a versioned plan with each artifact and enforce it across batches and resume.

Start with JSON encoding of whole annotated subtrees. Preserve legacy behavior for unannotated data.
Native map compilation and automatic storage selection are separate follow-ups.

The implementation is available on `codex/schema-json-storage`, based on `main` after parsing fix
PR #1046 merged. The sections below describe the design and its limits; the implementation status
records what was verified.

## Problem and motivating use case

The issue reports two storage failures already present in DD 0.9.3:

1. Arrow infers dynamic dictionaries as structs and adds absent keys as nulls.
2. Schema-valid heterogeneous arrays can fail checkpoint writing.

```text
Input:    {"fields": {"alpha": 1}}
          {"fields": {"beta": 2}}
Restored: {"fields": {"alpha": 1, "beta": null}}
          {"fields": {"alpha": null, "beta": 2}}

Mixed array: [1, "two", true, null]
Observed failure: ArrowInvalid while converting an element to the inferred numeric type.
```

Struct inference is reasonable for fixed records. It does not preserve dynamic dictionary semantics.
Accepting and validating JSON is currently separate from whether its inferred Parquet representation works.

Thom's motivating pipeline generates oracle tool-call sequences. Tool arguments contain strings, numbers,
booleans, arrays, and nested objects. He currently generates `arguments: dict[str, Any]`, then converts to a
persisted `arguments_json: str` with deterministic code. A homogeneous key/value-entry list does not solve
heterogeneous values. DD should offer an explicit equivalent without requiring storage-shaped LLM output.

Source discussion: [Thom's explanation](https://nvidia.slack.com/archives/C090G8EHX16/p1792423886449239).
His private source links have not been inspected; this proposal relies on his description and the issue.

The parsing data loss in #1043 is separate. PR #1046 preserves declared dictionary schemas; this work
depends on that fix or equivalent behavior and must not reintroduce its schema/pruning problems.

## Current implementation and affected boundaries

Code references below describe the worktree's starting commit, `89ddef27`.

| Location | Current behavior / required change |
| --- | --- |
| `packages/data-designer-config/src/data_designer/config/column_configs.py:320` | Converts Pydantic models to JSON Schema; retain the annotated declaration for config persistence and fingerprinting. |
| `packages/data-designer-config/src/data_designer/config/column_configs.py:667` | Custom columns have no general output-schema declaration; provide an explicit whole-column encoding route. |
| `packages/data-designer-engine/src/data_designer/engine/column_generators/utils/prompt_renderer.py:85` | Creates response recipes; supply the logical schema after storage annotations have been extracted. |
| `packages/data-designer-engine/src/data_designer/engine/processing/gsonschema/schema_transformers.py:12` | Bare-dictionary detection counts schema keys; metadata must not change object classification. |
| `packages/data-designer-engine/src/data_designer/engine/storage/artifact_storage.py:253` | Calls `DataFrame.to_parquet(index=False)` without an encoding plan. |
| `packages/data-designer-engine/src/data_designer/engine/dataset_builders/utils/row_group_buffer.py:104` | Writes checkpoint batches and moves them into the final dataset directory; checkpoint encoding is also generated-artifact encoding. |
| `packages/data-designer-config/src/data_designer/config/utils/io_helpers.py:112` | Reads/unifies physical schemas but has no logical decoding contract. |
| `packages/data-designer-engine/src/data_designer/engine/dataset_builders/utils/processor_runner.py:107` | After-generation processors reload final artifacts, process, and rewrite them. |
| `packages/data-designer/src/data_designer/interface/results.py:81` | Normal result loading delegates to artifact storage. |
| `packages/data-designer/src/data_designer/interface/results.py:239` | JSONL/CSV exports directly read Parquet; JSONL must restore objects rather than export escaped JSON strings. |
| `packages/data-designer/src/data_designer/interface/composite_workflow.py:867` | Workflow reads bypass artifact-storage loading. |
| `packages/data-designer-engine/src/data_designer/engine/resources/seed_reader.py:318` | Seed loading uses query-based readers; workflow handoffs need decoding after projection/batch materialization without losing encoding metadata. |
| `packages/data-designer-engine/src/data_designer/engine/dataset_builders/dataset_builder.py:646` | Resume compatibility must additionally compare the storage contract. |

Profiling also reloads artifacts through `ArtifactStorage.load_dataset_with_dropped_columns()` in
`packages/data-designer/src/data_designer/interface/data_designer.py`.

Resume progress recovery primarily uses completed files and row counts
(`dataset_builder.py:620`); it does not require decoding every completed row. The main requirements are a
stable encoding contract for new writes and correct logical reads wherever saved values are consumed.

## Goals

- Preserve annotated JSON subtree contents, including absent dictionary keys, explicit nulls, empty
  objects/arrays, heterogeneous arrays, and nested values.
- Keep LLM-facing schemas, validation, callbacks, and in-memory pipeline values natural.
- Make encoding explicit and deterministic before the first batch; never switch it based on observations.
- Use a common encoding contract for structured and custom-column outputs.
- Preserve existing unannotated artifacts and runs by default.
- Support result loading, processors, profiling, exports, and DD-to-DD workflow handoffs.
- Maintain interface -> engine -> config dependency direction and lazy heavy imports.

## Non-goals for the first release

- Automatically solving storage for every valid JSON Schema or every Python object.
- Automatically converting dictionaries to Parquet maps or changing unannotated column representations.
- Detecting JSON text by its contents, inferring encodings from the first non-null batch, or silently
  falling back to JSON after an Arrow failure.
- Providing arbitrary remote-reference resolution, conditional branch-dependent encodings, or schema migration.
- Replacing Parquet checkpoints with JSONL or changing the parser's pruning/union policies.
- Guaranteeing byte-identical JSON text, dictionary ordering, or arbitrary cross-language numeric precision.

Native fixed-record handling retains its existing missing-field normalization. Strict dictionary-key
presence preservation is promised inside explicitly JSON-encoded subtrees, not for all unannotated data.

## Proposed authoring API

### Schema-level declaration

```python
from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, JsonValue] = Field(
        json_schema_extra={"x-data-designer-storage": "json"}
    )


class Oracle(BaseModel):
    calls: list[ToolCall]
```

Use `Oracle` as the existing structured column's `output_format`. Direct JSON Schema users put the same
flat keyword on the corresponding schema node. A root annotation encodes the entire column. An annotation
on `arguments` encodes the entire value at that node, including everything nested beneath it.

The only initial value is `"json"`. Unknown values, malformed declarations, overlapping declarations,
and unsupported placements fail configuration/readiness checks with the column and schema location.
Unannotated schemas keep legacy storage behavior; no global RunConfig opt-in is required.

### Custom columns and other schema-less outputs

The optional `storage_encoding: Literal["json"] | None = None` on `SingleColumnConfig` is interpreted
as whole-column encoding. It lets custom
generators return JSON-compatible objects without introducing a new validation-schema API in this work.
The same internal plan handles root schema annotations and this column setting. Reject overlapping
whole-column and nested declarations initially rather than silently choosing precedence.

This route covers custom-column examples in #1045. A structured-only implementation must be described as
partial coverage of the issue. Per-field custom schemas can be considered separately.

## Compile a logical schema and a storage plan

1. Normalize a Pydantic model using public `model_json_schema()` APIs. Preserve the original annotated
   declaration in the saved configuration and its identity.
2. Traverse recognized schema positions, not arbitrary dictionaries inside `const`, `enum`, examples,
   defaults, or annotation values. Validate every DD storage declaration encountered.
3. Resolve supported local references and collect encoding operations at their instance locations.
   Reference-site annotations are meaningful even when the node is just `$ref` plus metadata.
4. Stop descending for encoding purposes at an annotated node: its whole JSON value is one storage unit.
   Detect/reject overlapping storage declarations during schema preflight.
5. Return a generation/validation schema with only the DD storage keyword removed. Preserve ordinary
   metadata and every validation constraint. Providers and prompt recipes see this cleaned schema.
6. Produce a deterministic, versioned plan keyed by output column and structural traversal operations.
   Use emitted property names/aliases, not Python attribute names or generated `$defs` naming conventions.

For `oracle: Oracle`, the plan means: select column `oracle`, property `calls`, each list item, property
`arguments`, then apply codec `json-v1`. Represent properties and list traversal with distinct typed
operations; do not use ambiguous dotted paths or unrestricted string wildcard matching.

### Supported schema subset

| Construct | First-release contract |
| --- | --- |
| Fixed properties and homogeneous array `items` | Traverse to any finite nesting depth; do not hardcode a maximum number of levels. |
| Local `$ref` / `$defs` | Resolve shared definitions and collect each reachable instance location. |
| Nullable annotated field | Put the annotation on the enclosing field; apply the same codec to all non-null values. |
| Entire annotated union, object, or array | Encode the complete selected node; its internal branches need no physical schema compilation. |
| Branch-specific declarations in `anyOf` / `oneOf`, `allOf`, or conditionals | Reject initially; request an annotation on a supported enclosing field/root. |
| Annotation beneath a dynamic dictionary | Require JSON encoding on the containing dictionary; native struct inference would otherwise still change its dynamic keys. Map-aware traversal is a follow-up. |
| Recursive references outside an encoded ancestor | Detect cycles. Reject plans requiring unbounded recursive traversal initially; an enclosing/root JSON declaration is supported. |
| External/dynamic refs or tuple-specific traversal | Reject when needed to locate storage declarations; do not silently skip affected annotations. |

Distinguish arbitrary runtime JSON depth below an annotated field from arbitrary JSON Schema feature
support. The former uses the JSON codec directly; the latter is deliberately bounded. Normal practical
serializer recursion/resource limits still apply. Unsupported cases get errors before generation.

## Encoding and decoding behavior

`json-v1` accepts JSON-compatible values: string-keyed objects, lists, strings, booleans, finite numbers,
and null. Reject non-string dictionary keys rather than silently stringifying them. Reject NaN/infinity,
unsupported Python objects, and implicit `default=str` conversions. Non-JSON types need an explicit
future codec or deterministic user normalization.

- Preserve `None` as physical null. Encode a literal string `"null"` as JSON string text, so it remains
  distinguishable from null on decoding.
- Preserve absent optional properties as absent while traversing; do not create an encoded placeholder.
  Any existing native parent-record normalization remains outside this codec's guarantee.
- JSON-encode every non-null value selected by the operation, including scalar strings, so mixed
  scalar/object values cannot produce incompatible encoded physical types.
- Force the declared physical leaf type even for null-only batches. Keep physical schemas consistent
  across a stage's batches; do not rely on values to infer the encoded leaf type.
- Encode a storage copy, leaving live buffers, nested dictionaries/lists, and caller-owned frames intact.
- Decode only declared fields, exactly once. Rewriting after a processor starts with logical values and
  re-encodes once. Never detect JSON by string contents or tolerate a malformed encoded value silently.

Wrap failures in project boundary errors with column, logical path, artifact, and row/batch context.
Avoid embedding full potentially sensitive tool arguments in error messages.

### Physical representation

Encoded leaves use explicit Arrow UTF-8 strings, including in empty and null-only batches. DD persists
its versioned encoding plan in Parquet schema metadata and run metadata. Generic Parquet readers receive
JSON text; DD readers restore logical values. This avoids requiring JSON extension-type support across
pandas and DuckDB. A future standard JSON logical annotation can be evaluated separately.

## Artifact contract and migration

Persist a storage-format version, canonical encoding operations, codec versions, and a fingerprint of
the encoding contract. Embed the relevant plan in each Parquet file and record the run/stage identity in
metadata before publishing the first batch. Reads should not require the original Pydantic model.

- Metadata absent: use legacy behavior. Ordinary external Parquet strings are not auto-decoded.
- Recognized metadata: validate it, then decode according to its version.
- Unknown versions, incompatible batch plans, or malformed declared JSON: fail with a clear error.
- Resume: compare the requested contract with the stored contract before scheduling new writes.
  Changed encoding cannot append to an existing run; require a fresh run or future explicit migration.
- Standalone files: carry the information needed to restore their values; a neighboring run manifest
  must not be the only source of decoding information.
- Merged/exported Parquet: explicitly preserve the encoding contract and verify input contracts match.
  Existing permissive schema-unification code must not choose one file's metadata arbitrarily.

Fingerprint the logical config and canonical storage plan separately. A missing legacy plan and an empty
new plan are compatible only under a documented version rule; do not invalidate all old resumes by default.
Batch publication and metadata persistence must remain safe across crashes between file writes and run
metadata updates. This work does not promise atomic replacement of all after-generation output files.

## Readers, processors, and stage handoffs

Introduce shared low-level decoding/metadata utilities in the config package's IO area (or a dedicated
config-side storage-contract module). Engine and interface callers can use them without reverse imports.
These helpers operate on explicit plans and values; config DTOs do not call the engine.

The engine owns schema compilation and write orchestration. ArtifactStorage accepts the compiled plan
and is the primary write boundary. Generic IO helpers recognize plans when reading DD artifacts.
Keep PyArrow/pandas imports lazy and avoid an extensible codec registry until another codec requires it.

Audit result reads, profiling, dropped-column loads, processor outputs, and composite-workflow helpers.
Seed readers also need the original file's contract: query engines may discard Parquet metadata or expose
JSON as text. Discover/validate the contract before projection, remap selected columns, and decode batches
after materialization. Fail explicitly on unsupported projections rather than silently handing strings to
the next stage. Test both file/directory seeds and DD-generated workflow handoffs.

After-generation processors read logical values and write through the same encoder. Main-dataset
processors must preserve the declared paths and compatible logical shapes; automatic path remapping is
outside this implementation. Separate schema-transform output datasets declare whole-column encoding
with `SchemaTransformProcessorConfig.json_storage_columns`. They do not inherit the main dataset plan.

Preview remains logical and should not encode/redecode merely to display data. Validate its configuration
with the same compiler so preview cannot accept a malformed storage declaration that create later rejects.

## Export and compatibility behavior

| Consumer | Contract |
| --- | --- |
| DD processors, profiling, result loading | Restored logical values. |
| JSONL export | Actual nested JSON with protected numbers preserved, not escaped strings in place of objects. |
| CSV export | Explicit JSON text for structured cells; quote/escape through the CSV writer. |
| Parquet export and published batch files | Documented native fields and JSON-encoded leaves with metadata. |
| External Parquet/Hub consumers | Reader-dependent representation; no universal automatic dictionary-restoration claim. |

No silent change to existing exports or default storage. Document loss of native nested predicate access
inside JSON-encoded subtrees, encoding/decoding CPU cost, and potential changes in compression/size.
External readers see UTF-8 JSON text. PyArrow/pandas/DuckDB paths are tested; live Hub publication and
independent external-engine compatibility have not been exercised.

## Dependency/version considerations

Current declarations are Pydantic `>=2.9.2,<3`, jsonschema `>=4,<5`, and PyArrow `>=24,<25`.
Pydantic 1 is outside scope. Use public Pydantic schema generation APIs and dict-valued
`json_schema_extra`. Prefer the flat storage keyword to avoid nested metadata overwrite during composition.

Adopt Draft 2020-12 for annotation-bearing schemas initially. The annotation compiler owns extraction;
it must not depend on version-specific annotation reporting or jsonschema validator internals. Unknown
keywords do not validate their own payload, so DD must check this keyword's value and placement itself.
Treat the JSON Schema draft and the Python jsonschema package version as separate compatibility axes.

Fix metadata-sensitive bare-dictionary classification. Removing the storage keyword must leave generation
and validation behavior equivalent to the otherwise identical unannotated logical schema.

## Delivery sequence

1. Review this plan, confirm the annotation name, custom-column entry point, schema subset, and external
   representation. Keep #1043's parser fix independent.
2. Add configuration/plan DTOs and compiler tests. Produce cleaned schemas without mutating declarations.
3. Add shared codec/metadata helpers and explicit physical schemas, with legacy behavior as the default.
4. Wire writers and every DD-controlled reader, including seed/workflow handoffs and exports. Do not ship
   encoding while known internal paths still return physical text unexpectedly.
5. Add resume/artifact compatibility checks, lifecycle tests, performance measurements, and user docs.
   Reconcile this plan with delivered behavior before considering implementation complete.

Each implementation PR should have a coherent boundary and describe feature availability accurately.
A compiler-only PR must not claim to fix #1045. Codec-only helpers should not expose a partially working
public option. No need to implement map inference or a complete schema-to-Arrow compiler first.

## Validation and acceptance criteria

- Configuration: Pydantic and direct schemas agree; root/field annotations, references, aliases, nullable
  fields, nested lists, invalid values, overlaps, unsupported constructs, and recursive reference checks.
- Schema isolation: no storage metadata reaches model/provider schemas; constraints and unrelated metadata
  survive; annotations inside literal data are untouched; schema objects are not mutated.
- Codec: absent keys vs explicit null, empty values, Unicode/escaping, booleans, mixed arrays, nested JSON,
  string `"null"` vs null, already-JSON-looking strings, large integers, finite floats, and rejected objects.
- Physical files: stable encoded leaf types for empty/null-only/mixed batches, nested JSON logical types,
  standalone-file decoding, metadata preservation, incompatible plan rejection, and legacy compatibility.
- Lifecycle: create, preview, after-generation rewriting, dropped columns, profiling, result loads,
  file/directory seeds, projected workflow handoffs, custom columns, JSONL/CSV/Parquet exports, and publication.
- Resume: unchanged policy succeeds, changed policy fails before generation, out-of-order completed files,
  and crash recovery around initial plan persistence and batch publication.
- Versions: supported minimum and current Pydantic/jsonschema combinations, plus the supported PyArrow
  range. Do not copy generated schema text as golden fixtures across versions; test semantics and plan output.
- Performance: compare representative tool-call payloads and native datasets for write/read throughput,
  memory, and file size. Establish a baseline before choosing a regression threshold.

Success for Thom's workflow means the LLM-facing `arguments` stays an object, artifacts accept its flexible
JSON contents, and normal DD reads/hand-offs restore those contents without pipeline-specific validators.
Unannotated behavior is unchanged; the opt-in feature does not make every schema automatically storable.

## Implementation status and remaining limits

Implemented the schema compiler, logical/physical codec, versioned contracts, artifact writes/reads,
resume policy checking, seed/workflow handoffs, processor reads and outputs, result exports, and user docs.
Config fingerprints preserve the previous identity when the new settings retain their defaults.
Artifact writes use a temporary file and atomic replacement so failed serialization preserves prior files.

Thom clarified that he wants preflight validation to avoid spending on LLM calls before discovering
storage issues ([clarification](https://nvidia.slack.com/archives/C090G8EHX16/p1792437437428629)). The
existing engine config validator now checks structured output schemas before model health checks or
generation. Required, unconditional empty objects and unsupported constant values are errors. Optional
or nullable occurrences, dynamic dictionaries, mixed types, recursive references, and other uncertain
schema patterns produce actionable warnings with column and schema location. Encoded subtrees are
excluded; unrelated unencoded fields are still checked. Numeric promotion and nullable scalar unions
do not warn. Backend probes for enum constants are bounded, and a partial probe cannot justify an error.
Multiple enum values only warn even if they each fail alone: their struct fields may complement one
another when inferred together. Only a single unsupported constant justifies a fatal literal finding.

This is conservative analysis of known patterns, not a full schema satisfiability or storage proof.
Custom columns without output schemas retain runtime checks. Unsupported native writes still raise a
project error identifying the column and suggesting JSON storage. Successful unannotated struct
normalization is unchanged. Invalid storage annotations themselves fail config validation.

Verified full config, engine, and interface suites, plus targeted feature tests. Loopback socket tests need
execution outside the sandbox; their complete test files passed on rerun. The targeted final run covers
metadata/version/dialect rejection and failed-write preservation. Ruff and internal Fern links also pass.

After adding preflight validation, full suites passed with 2,637 engine tests and 1,924 combined config
and interface tests (3 skips). The final preflight suite passed 51 cases, including the complementary-enum
regression added after that full run. Tests verify diagnostics and failure before the model readiness
gate. Before opening the draft PR, the branch was rebased onto `main` after #1046 merged. The final
full suites passed with 2,655 engine tests and 1,944 combined config/interface tests (3 skips). Four
JSONL regression cases additionally verify exact float/large-integer preservation and null/date handling.
Ruff and formatting pass on all changed Python files. The pinned Fern CLI check passed with no errors;
its remote redirects check was skipped because Fern authentication was unavailable. The deterministic
internal-link check passed across 90 pages.

Remaining compatibility work before release:

- This is a finite traversal compiler, not a complete JSON Schema to Arrow compiler. Branch-dependent,
  dynamic dictionary, and recursive traversal require encoding an enclosing node. Arbitrary JSON depth
  inside that encoded node is supported within Python's usual recursion limits.
- Main-dataset processors must preserve encoded paths. Automatic output-path remapping is not provided.
- Dependency checks used the current environment. Minimum-version matrix checks and representative
  performance benchmarks remain follow-ups; no cross-version or throughput guarantee is claimed.
- Live Hub publication and independent external-engine interpretation have not been exercised.
- Review the public annotation/config names and metadata migration promise before release.

## Post-draft review hardening

Three independent reviewers each performed ten substantive passes per review cycle. The first two
cycles found and reproduced edge cases using real codec files and public pipeline runs; fixes were
committed with regression tests before another stable review cycle.

- Canonicalize path operations, reject contradictory/ambiguous container traversal, and validate
  physical metadata paths and string leaves before reading any rows.
- Resolve local URI-fragment pointers correctly, including escaped tokens and array indices; reject
  nested resource traversal and keep unsupported preflight dialects/scopes warning-only.
- Match constant probes to the actual pandas writer, ignore inactive type-specific schema keywords,
  respect positional array limits, discarded-column policy, conditional generation, and processor
  transformations when determining whether a storage error is proven.
- Preserve nullable and projected DataFrame/file seeds, carry metadata through `from_dataframe`, and
  register combined Arrow tables for annotated DataFrame seeds to avoid nested multichunk data loss.
- Restore partitioned directory reads, normalize overflow and malformed-metadata recursion errors,
  preserve native null semantics, and keep CSV structured cells valid JSON without rounding protected
  numbers.
- Reject impossible literal processor output-storage names before generation while leaving dynamic
  template-key checks conservative. Keep similarly named plugin fields opaque to fingerprint rules.

The first hardened candidate passed 2,696 engine tests and 2,010 config/interface tests (3 skips).
The full config suite also passed 803 tests with Pydantic 2.9.2 and jsonschema 4.0.0 in an isolated
environment. The engine cannot import with that older Pydantic because of an unchanged telemetry
class-variable incompatibility; this review did not modify that unrelated subsystem. Feature lifecycle
tests with the current Pydantic and minimum jsonschema passed separately.

## References

- [Storage issue #1045](https://github.com/NVIDIA-NeMo/DataDesigner/issues/1045)
- [Parsing fix PR #1046](https://github.com/NVIDIA-NeMo/DataDesigner/pull/1046)
- [Pydantic schema metadata and merging](https://docs.pydantic.dev/latest/concepts/json_schema/#using-json_schema_extra)
- [JSON Schema reference behavior](https://json-schema.org/understanding-json-schema/structuring#ref)
- [Arrow JSON extension type](https://arrow.apache.org/docs/python/generated/pyarrow.json_.html)
- [Parquet JSON logical type](https://parquet.apache.org/docs/file-format/types/logicaltypes/#json)
