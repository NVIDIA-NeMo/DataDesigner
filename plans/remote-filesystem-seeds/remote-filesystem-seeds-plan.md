---
date: 2026-06-04
authors:
  - mknepper
---

# Plan: Remote filesystem-backed seed sources for Data Designer

## Summary

When the Data Designer NeMo Platform service runs in **remote execution mode**,
the `data_designer_nemo` plugin only accepts two seed source types:
HuggingFace (`seed_type=hf`) and the Files service (`seed_type=nmp`). Locally,
the upstream Data Designer library additionally supports `directory`,
`file_contents`, and `agent_rollout` seed sources that read files from local
disk.

This RFC proposes bridging that gap by making the upstream library's
filesystem layer **injectable** via a generic `FileSystemProvider` seam, and
having the platform plugin inject a provider backed by the existing
`FilesetFileSystem` (an fsspec `AsyncFileSystem`). The result: `directory` and
`file_contents` seed sources work remotely, reading their files from the
Files service instead of local disk — with the upstream library remaining
completely NeMo Platform-agnostic.

`agent_rollout` is deliberately descoped from this plan. It is the one
filesystem seed source that is not immediately ready for an injectable
filesystem layer because its format handlers bypass fsspec and open
`pathlib.Path` objects directly. Agent rollout support may move out of the core
library and become a plugin instead, and there are open questions on the
nemo-platform side about getting agent session data into the Files service,
so this plan avoids adding new core AgentRollout reader abstractions now.

## Problem

### Current state

`RemoteDataDesignerContext.get_seed_readers()`
(`nemo-platform/packages/data_designer_nemo/src/data_designer_nemo/context.py:143`) returns
only:

```python
[
    HuggingFaceSeedReader(),
    FilesetFileSeedReader(self._sdk),
]
```

`LocalDataDesignerContext.get_seed_readers()` returns those plus
`LocalFileSeedReader`, `DataFrameSeedReader`, `DirectorySeedReader`,
`FileContentsSeedReader`, and `AgentRolloutSeedReader`.

The remote restriction is enforced at parse time by
`unsupported_features.py`, where `_SUPPORTED_SEED_TYPES = {"hf", "nmp"}`.

### Why the in-scope types are local-only

`DirectorySeedReader` and `FileContentsSeedReader` descend from the upstream
`FileSystemSeedReader` (`data-designer-engine/.../seed_reader.py:376`). They
enumerate and read files under a configured directory. The library assumes that
directory lives on the **local disk of the process that runs the workflow**. In
remote mode the process runs inside the service, so the user's files are not
there — they live in the Files service.

There are exactly three in-scope places where the upstream code hardcodes
"local disk":

| # | Coupling point | Location | Difficulty |
|---|---|---|---|
| 1 | `create_filesystem_context()` builds `DirFileSystem(fs=LocalFileSystem())` | `seed_reader.py:245-249` | Easy |
| 2 | `SeedReaderFileSystemContext.root_path: Path` is a concrete local path | `seed_reader.py:58` | Medium |
| 3 | Config validator calls `path.is_dir()` against local disk at parse time | `seed_source.py:205-211, 134-138` | Easy |

The important good news: `DirectorySeedReader` and `FileContentsSeedReader`
**already perform all I/O through `context.fs`** (e.g. `context.fs.open(...)`
at `seed_reader.py:578`). And `FilesetFileSystem`
(`nemo-platform/packages/filesets/.../filesystem.py:300`) is already a working fsspec
`AsyncFileSystem`. The plumbing to connect them exists; it is only blocked by
the hardcoded `LocalFileSystem()` in coupling point #1.

The excluded `AgentRolloutSeedReader` is the sole exception in the broader
`FileSystemSeedReader` family: its handlers receive a `pathlib.Path` and open
files directly, never touching `context.fs`. Refactoring that path is now out of
scope for this bridge.

## End-user interface

Users build configs the same way they do today, via
`DataDesignerConfigBuilder.with_seed_dataset(<seed source>)`. Remote mode does
**not** introduce new seed source classes — it reuses the upstream
`DirectorySeedSource` and `FileContentsSeedSource`. The only thing that changes
is what string the user puts in `path`.

### Path grammar

In remote mode the `path` field carries the **same fileset reference grammar**
as `FilesetFileSeedSource`:

```
<workspace>/<fileset>#<path-fragment>
```

- `<workspace>/` is optional; when omitted it is inferred from the current
  context (the same workspace inference used by `FilesetFileSeedReader`, see
  `fileset_file_seed_reader.py:42-53`).
- The `#` delimiter separates the **fileset** from the **path fragment within
  it**.
- The semantics of the fragment differ by reader, mirroring the local library:
  - For `FilesetFileSeedSource` (`nmp`) the fragment points at a **file** (or
    wildcard) read tabularly by duckdb.
  - For `DirectorySeedSource` / `FileContentsSeedSource` the fragment is the
    **root directory** the reader enumerates under. `file_pattern` and
    `recursive` then operate within that directory, exactly as they do on local
    disk.

This means a user who already knows how to reference a fileset for `nmp` does
not learn a new grammar — they reuse the same ref and additionally set the
directory-style knobs (`file_pattern`, `recursive`, and per-type fields).

### Worked examples (remote mode)

Existing remote tabular reader (unchanged), for contrast:

```python
# Reads a single parquet/csv file (or wildcard) as a table via duckdb.
builder.with_seed_dataset(
    FilesetFileSeedSource(path="default/my-seeds#data/train.parquet")
)
```

New, enabled by this RFC:

```python
# directory: manifest of files (one row per file) under data/traces/
builder.with_seed_dataset(
    DirectorySeedSource(
        path="default/my-seeds#data/traces",
        file_pattern="*.jsonl",
        recursive=True,
    )
)

# file_contents: one row per file, each file's text in a `content` column
builder.with_seed_dataset(
    FileContentsSeedSource(
        path="default/my-docs#corpus",
        file_pattern="*.md",
        recursive=True,
        encoding="utf-8",
    )
)
```

### Notes and caveats for the interface

- **`file_pattern` matches basenames only** (not relative paths) — same rule as
  local, enforced upstream by `_validate_filesystem_seed_source_file_pattern`
  (`seed_source.py:214-221`).
- **Workspace omission** is allowed (`my-seeds#data/traces`); the provider fills
  it from context. Including it (`default/my-seeds#data/traces`) is explicit and
  always safe.
- The `#` fragment is overloaded: "a file" for `nmp` vs "a directory" for the
  filesystem readers. This is intentional and matches the local split between
  `LocalFileSeedSource` (a file) and the `FileSystemSeedSource` family (a
  directory).

## Goals

- Make `directory` and `file_contents` seed sources usable in remote execution
  against files stored in the Files service.
- Keep the upstream Data Designer library **NeMo-agnostic**. It may expose
  generic protocols and accept dependencies via injection; it must not know
  about NeMo Platform, the SDK, or Filesets.
- Preserve all existing behavior: local execution unchanged, the existing
  `nmp`/`FilesetFileSeedReader` duckdb path unchanged.

## Non-goals

- Replacing or deprecating `FilesetFileSeedReader` / `seed_type=nmp`. That
  duckdb-based tabular reader (single file or wildcard parquet) is
  complementary and stays.
- Optimizing per-file read concurrency for large directories (noted as future
  work).
- Adding new seed source *config* types. The `directory` and `file_contents`
  config classes already exist upstream and are serializable over the wire.
- Adding remote `agent_rollout` support. Agent rollout's direct `pathlib` file
  access and local-default path semantics make it a larger change, and the type
  may move out of the core library into a plugin.

## Design

### Core idea

`FileSystemSeedReader` already funnels file discovery and reads through an
abstract `SeedReaderFileSystemContext(fs, root_path)`. Everything downstream
(`build_manifest`, `get_matching_relative_paths`, `hydrate_row`) consumes
`context.fs` and `context.root_path` abstractly. If we make **which filesystem
backs that context** injectable, then `directory` and `file_contents` work
remotely with no reader changes.

The injection follows the pattern already used in this plugin:
`FilesetFileSeedReader(self._sdk)` — readers accept their NeMo dependency via
constructor.

### Part A — Upstream library (NeMo-agnostic)

#### A1. Introduce a `FileSystemProvider` protocol

Add a generic seam in `data_designer.engine.resources`:

```python
class FileSystemProvider(Protocol):
    """Resolves a seed source's runtime path into a rooted fsspec filesystem.

    Implementations decide what backing filesystem to use. The default uses the
    local disk; hosts can inject one backed by a remote object store, a fileset
    service, etc.
    """
    def create_context(self, *, runtime_path: str) -> SeedReaderFileSystemContext: ...

    # Authors the precise, backend-specific existence error. See A3 Layer 2.
    def ensure_root_exists(self, *, runtime_path: str) -> None: ...
```

- The **default implementation** reproduces today's behavior:
  `DirFileSystem(path=resolved, fs=LocalFileSystem())`, with `ensure_root_exists`
  performing the local `is_dir()` check (see A3 Layer 2 for the full rationale
  and the per-backend message wording).
- `FileSystemSeedReader` accepts an optional `fs_provider` in its constructor,
  defaulting to the local provider. `create_filesystem_context()` delegates to
  it, and `_get_filesystem_context()` calls `ensure_root_exists()` before
  building the context.

This single seam unlocks `directory` + `file_contents` remotely with **zero
changes to the reader subclasses**.

#### A2. Make `root_path` filesystem-agnostic

`SeedReaderFileSystemContext.root_path: Path` is used for display/metadata (the
`source_path` column and error messages). That value does not need to be an
openable local filesystem path for `directory` or `file_contents`; reads already
go through `context.fs`.

Loosen `root_path` so it can carry a displayable remote root, such as a fileset
ref rendered for traceability. Keep it for metadata only; new read paths should
continue to use `context.fs`.

#### A3. Defer config-parse-time local validation (two-layer validation)

The upstream config classes bake **two** local-disk assumptions into the Pydantic
model itself, and both fire at *construction time* — client-side, before any SDK,
provider, or reader exists, and before the config is serialized over the wire:

1. `validate_path` field validator (`seed_source.py:134-138`) →
   `_validate_filesystem_seed_source_path` → `path.is_dir()`
   (`seed_source.py:205-211`). This raises the moment a user constructs
   `DirectorySeedSource(path="default/my-seeds#data/traces")`, because the ref
   is not a local directory.
2. `model_post_init` / `runtime_path` (`seed_source.py:140-148`) →
   `_resolve_filesystem_runtime_path` → `Path(path).expanduser().resolve()`
   (`seed_source.py:176-177`). Even if validation passed, this *mangles* the
   ref: `Path("default/my-seeds#data/traces").resolve()` yields something like
   `/cwd/default/my-seeds#data/traces`, destroying the `#` and workspace
   semantics.

Because validation runs at construction with no filesystem in scope, the config
layer can only do **structural** checks. The fix is to split validation into
two layers:

**Layer 1 — upstream config (Pydantic): structural only.**

- `validate_path` drops `is_dir()` and keeps only cheap structural checks
  (non-empty, etc.). It no longer asserts existence.
- For the in-scope sources, `runtime_path` stops calling
  `Path(...).resolve()` at construction time and preserves the user-authored
  path string for the provider to interpret. This is deliberately generic: the
  core config package does **not** parse, classify, or otherwise know about
  NeMo Platform fileset refs or any other host-specific remote path grammar.
- Net effect: `DirectorySeedSource(path="default/my-seeds#data/traces")`
  constructs successfully. **Same classes work in both local and remote modes.**

> **Scope note — deliberate asymmetry.** This deferral applies only to the
> in-scope `FileSystemSeedSource` types (`directory` / `file_contents`), which
> the remote bridge needs. `LocalFileSeedSource` keeps its eager
> construction-time `is_file()` validation; it is *not* deferred, because
> `local` is unsupported in remote mode regardless, so deferring it unlocks
> nothing (see "Server-side request validation"). The resulting eager-vs-lazy
> asymmetry between the single-file and directory-family sources is intentional,
> not an oversight.

The current validator is shared by `FileSystemSeedSource`, so the implementation
must preserve this boundary explicitly. If needed, split or override validation
so only `DirectorySeedSource` and `FileContentsSeedSource` get provider-owned
raw-path handling; `AgentRolloutSeedSource` should keep its current local-path
behavior until its ownership and plugin direction are decided.

**Accepted local relative-path semantics change.** Today a local relative path
is anchored when the config object is constructed because `model_post_init`
caches an absolute `_runtime_path`. This plan intentionally changes that for
`DirectorySeedSource` and `FileContentsSeedSource`: relative local paths are
resolved by the active filesystem provider at validate/read time. That means a
script that constructs `DirectorySeedSource(path="./seeds")`, changes cwd, and
then validates/reads will resolve `./seeds` against the later cwd. This is an
accepted tradeoff because it keeps config objects declarative and portable when
serialized/shared, and avoids adding hidden construction-cwd state that generic
core helpers could accidentally misuse in remote contexts. Users who need stable
local anchoring can opt in explicitly by passing an absolute path, e.g.
`DirectorySeedSource(path=str(Path("./seeds").resolve()))`. This behavior change
must be documented in the upstream changelog and covered by tests.

**Layer 2 — filesystem-aware existence check, in the provider/reader.**

The "does this actually exist?" check moves down to where a filesystem is known
— the `FileSystemProvider` / `FileSystemSeedReader` — rather than being deleted.
This is the critical correction: simply removing the Pydantic `is_dir()` check
must **not** silently drop existence validation for anyone, including plain
upstream library users.

This works because existence validation is already reachable from every
library `validate` entry point:

- The upstream `DataDesigner.validate()`
  (`interface/data_designer.py:535`) and the CLI both delegate to
  `compile_data_designer_config(...)`, which calls
  `_resolve_and_add_seed_columns` → `seed_reader.get_column_names()`
  (`engine/compiler.py:20-34`). For a `FileSystemSeedReader`,
  `get_column_names()` builds the manifest, which calls
  `context.fs.find(...)` (`seed_reader.py:262`). So upstream `validate()`
  **already touches the filesystem** via the provider — existence is exercised
  there with no workload started.
- The NeMo `RemoteDataDesignerContext.validate` additionally calls
  `validate_seed` (`context.py:133`), which already checks fileset existence and
  permissions for `FilesetFileSeedSource` (`seed.py:28-40`).

**Avoiding a UX regression: existence must be checked, not *discovered*.**

The risk with deferral is not "no error" — it is *degraded* error quality.
Today a bad path produces a typed, specific error at construction. If we simply
delete the Pydantic checks and let `validate()` → `get_column_names()` →
`build_manifest()` stumble into the problem, the failure surfaces as a
downstream *symptom* rather than a clear existence error. The degradation is
graded:

| Failure | Today (Pydantic, at construction) | Naive deferral (incidental discovery) | Quality |
|---|---|---|---|
| Directory does not exist | `InvalidFilePathError: "🛑 Path X is not a directory"` | `DirFileSystem.find("")` returns `[]` → `SeedReaderError("No files matched file_pattern ... under X")` (`seed_reader.py:271-273`) | **Misleading** — implies the dir exists but is empty |
| Directory exists, no matches | (not checked) | same "No files matched ..." | Fine — accurate |
| Bad extension | `InvalidFileFormatError` | unchanged (stays structural, see below) | Fine |

So the regression is concentrated in the missing-directory case. The design must
satisfy **four requirements** so `validate()` UX is not degraded:

1. **Explicit existence preflight, not incidental discovery.** Existence must be
   asserted as a first-class check that runs *before* manifest building /
   `find()` — i.e. in `FileSystemSeedReader` when it creates the filesystem
   context (`_get_filesystem_context`, before `build_manifest`). The error must
   describe *existence*, not a downstream consequence.
2. **Precise, per-backend, user-facing message.** Each backend, via its
   provider, emits a tailored message at least as good as today's, and distinct
   from the "exists but empty" case:
   - **Local provider** → `"🛑 Seed source directory '<path>' does not exist."`
     (matches today's intent).
   - **Fileset provider** → `"fileset '<ws>/<fs>' not found"` /
     `"path '<fragment>' not found in fileset '<ws>/<fs>'"` — *better* than
     today, which could not express this at all.
3. **Stable error *classification* at the `validate()` boundary.** Today these
   are `InvalidFilePathError` (a `config.errors` type); relocated, they become
   `SeedReaderError` (an `engine` type). Anything that catches
   `InvalidConfigError` to render a friendly "your config is wrong" message
   (the upstream CLI; NeMo's `validate()` translating into `NDDError`) must not
   start treating this as an unexpected internal crash. The readers may
   legitimately raise `SeedReaderError`, but the **validate boundary is
   responsible for classifying it as a config/user error**, not an internal
   error. On the NeMo side this mirrors what `validate_seed` already does —
   translating SDK `NotFoundError` → `NDDInvalidConfigError` (`seed.py:32-40`).
   The upstream side must ensure `compile_data_designer_config` /
   `DataDesigner.validate` surface a `SeedReaderError` from existence checking
   as a config violation, not a traceback.
4. **Keep cheap structural checks in Pydantic.** Only the *filesystem-touching*
   check is deferred, and only for `DirectorySeedSource` /
   `FileContentsSeedSource`.
   Non-filesystem validation stays at construction so most malformed configs
   still fail fast even without calling `validate()`: non-empty `path` and valid
   `file_pattern` shape (`seed_source.py:214-221`).

Together these mean upstream `DataDesigner.validate()` and the CLI keep a clear,
typed, actionable error — sourced from the reader/provider instead of the
Pydantic model — on every `validate()` path, without the config layer assuming
local disk.

- **Remote (NeMo):** additionally extend `validate_seed` to recognize the two
  in-scope filesystem source types, parse their `#`-ref, and verify the fileset
  exists via the SDK for an early, network-cheap check (see B3). The
  provider-level preflight (above) remains the backstop reached through
  `get_column_names()`.

**Where the messages live: an `ensure_root_exists` provider seam.**

The check is invoked in **one place upstream**, but the *wording* is authored by
**whichever provider is injected**. Upstream owns "an existence check happens
here"; the provider owns "here's what to say when it fails." This is what makes
the messages precise *and* keeps the library NeMo-agnostic.

The `FileSystemProvider` protocol (A1) gains one method for this:

```python
# data_designer/engine/resources/seed_reader.py  (upstream)

class FileSystemProvider(Protocol):
    def create_context(self, *, runtime_path: str) -> SeedReaderFileSystemContext: ...
    def ensure_root_exists(self, *, runtime_path: str) -> None: ...


class LocalFileSystemProvider:
    """Default provider — today's behavior, plus an explicit existence check."""

    def create_context(self, *, runtime_path: str) -> SeedReaderFileSystemContext:
        resolved = Path(runtime_path).expanduser().resolve()
        rooted = DirFileSystem(path=str(resolved), fs=LocalFileSystem())
        return SeedReaderFileSystemContext(fs=rooted, root_path=resolved)

    def ensure_root_exists(self, *, runtime_path: str) -> None:
        resolved = Path(runtime_path).expanduser().resolve()
        if not resolved.is_dir():
            # The message that used to live in the Pydantic validator
            # (_validate_filesystem_seed_source_path), relocated verbatim.
            raise SeedReaderError(f"🛑 Seed source directory '{resolved}' does not exist.")
```

The reader invokes the preflight at the single choke point every read funnels
through — `_get_filesystem_context` (`seed_reader.py:494`) — *before*
`build_manifest()` / `find()` can degrade the failure into "no files matched":

```python
# FileSystemSeedReader._get_filesystem_context  (upstream)

def _get_filesystem_context(self) -> SeedReaderFileSystemContext:
    self._ensure_attached()
    context = getattr(self, "_filesystem_context", None)
    if context is None:
        self._fs_provider.ensure_root_exists(runtime_path=self.source.runtime_path)  # NEW
        context = self._fs_provider.create_context(runtime_path=self.source.runtime_path)
        self._filesystem_context = context
    return context
```

The NeMo provider implements the same seam with fileset-flavored wording,
translating the low-level `FileNotFoundError` that `FilesetFileSystem._info`
raises (`filesystem.py:503-526`) into a friendly `SeedReaderError`:

```python
# data_designer_nemo/fileset_filesystem_provider.py  (NeMo side)

class FilesetFileSystemProvider:
    def __init__(
        self,
        sdk: NeMoPlatform | AsyncNeMoPlatform,
        validated_roots: set[str] | None = None,
    ):
        if isinstance(sdk, AsyncNeMoPlatform):
            sdk = async_to_sync_sdk(sdk)
        self._sdk = sdk
        self._validated_roots = set() if validated_roots is None else validated_roots

    def create_context(self, *, runtime_path: str) -> SeedReaderFileSystemContext:
        fs = FilesetFileSystem(self._sdk)
        root = build_fileset_ref(runtime_path, ...)        # "<ws>/<fs>#<fragment>"
        rooted = DirFileSystem(path=root, fs=fs)
        return SeedReaderFileSystemContext(fs=rooted, root_path=PurePosixPath(root))

    def ensure_root_exists(self, *, runtime_path: str) -> None:
        workspace, fileset, fragment = parse_fileset_ref(runtime_path, ...)
        ref = build_fileset_ref(runtime_path, ...)
        if ref in self._validated_roots:
            return

        fs = FilesetFileSystem(self._sdk)
        # fsspec sync exists() facade; FilesetFileSystem._info raises
        # FileNotFoundError for a missing path.
        if not fs.exists(ref):
            if not fs.exists(f"{workspace}/{fileset}"):
                raise SeedReaderError(f"🛑 Fileset '{workspace}/{fileset}' not found.")
            raise SeedReaderError(
                f"🛑 Path '{fragment}' not found in fileset '{workspace}/{fileset}'."
            )
```

So `directory` / `file_contents` running remotely get the fileset-specific
message, while the local default provider keeps today's "directory does not
exist" wording. (`LocalFileSeedSource` is untouched — it retains its eager
construction-time `is_file()` check; see the A3 scope note.)

Two implementation notes:

- **Avoid a double round trip with a per-context validated-root cache.**
  `ensure_root_exists` is free locally (`is_dir`) but a network call remotely
  (`fs.exists`). For remote, `RemoteDataDesignerContext` owns a
  request-scoped `set[str]` of canonical fileset root refs that successfully
  passed `validate_seed`. It passes the same set into
  `FilesetFileSystemProvider`, and `ensure_root_exists` returns immediately
  when the canonical root is present. If a caller skips `validate()` or creates
  a provider in a different context, the set is empty and the provider preflight
  still runs as the lazy backstop.
- **One error class crosses the boundary.** The provider must catch low-level
  `FileNotFoundError` (`filesystem.py:503`) and re-raise as `SeedReaderError`
  with the friendly text, so the reader and the `validate()` classification
  boundary (requirement 3) only ever deal with `SeedReaderError`.

**Behavior change being accepted.** With the four requirements above, error
*quality* is preserved (arguably improved for filesets) on every `validate()`
path. There are two accepted local behavior changes:

1. A missing directory now errors when `validate()` (or a read) runs rather than
   at the instant `DirectorySeedSource(...)` or `FileContentsSeedSource(...)` is
   constructed. Any code path that calls `validate()` — the high-level
   `DataDesigner.validate`, the CLI, and the platform preview/job flows — still
   gets a precise, typed pre-flight error with **no wasted workload**. The only
   path that loses an early signal is constructing the source in a script that
   never validates and never reads; even there, the cheap structural checks
   (requirement 4) still fire at construction.
2. Relative local paths are no longer anchored to the cwd at config construction
   time. They are resolved by the local filesystem provider at validate/read
   time, while remote providers interpret the same raw `path` string according
   to their own grammar. This makes serialized configs more portable and keeps
   provider-specific path interpretation out of core. Users who need the old
   stable anchoring behavior can pass an absolute path explicitly.

Given directories can be created/deleted between construction and run, and
relative-path portability is valuable for shared configs, these residual gaps
are acceptable, but must be documented in the upstream changelog.

### Part B — NeMo side (`data_designer_nemo`)

#### B1. A Fileset-backed `FileSystemProvider`

Implement `FilesetFileSystemProvider` as the complete provider shown in A3
(`create_context` plus `ensure_root_exists`). The NeMo-side provider must
satisfy the upstream `FileSystemProvider` protocol, so B1 only calls out the
NeMo-specific wiring details:

- Normalize `AsyncNeMoPlatform` to sync via `async_to_sync_sdk` before creating
  `FilesetFileSystem`.
- Reuse the workspace-prefixing behavior from
  `fileset_file_seed_reader.py:42-53`.
- Use the `build_fileset_ref` helper from `filesystem.py`.
- Accept the request-scoped `validated_roots` set from
  `RemoteDataDesignerContext` to skip duplicate existence checks after
  `validate_seed`.

#### B2. Wire it into `RemoteDataDesignerContext.get_seed_readers()`

```python
class RemoteDataDesignerContext:
    def __init__(self, sdk: AsyncNeMoPlatform | NeMoPlatform, workspace: str):
        self._sdk = sdk
        self._workspace = workspace
        self._validated_filesystem_roots: set[str] = set()

    async def validate(self, config: dd.DataDesignerConfig) -> list[NDDError]:
        sdk = self._async_sdk()
        errors: list[NDDError] = []
        ...
        try:
            if validated_root := await validate_seed(config, self._workspace, sdk):
                self._validated_filesystem_roots.add(validated_root)
        except NDDError as e:
            errors.append(e)
        ...

    def get_seed_readers(self) -> list[SeedReader]:
        provider = FilesetFileSystemProvider(
            self._sdk,
            validated_roots=self._validated_filesystem_roots,
        )
        return [
            HuggingFaceSeedReader(),
            FilesetFileSeedReader(self._sdk),          # keep: single-file/duckdb path
            DirectorySeedReader(fs_provider=provider),
            FileContentsSeedReader(fs_provider=provider),
        ]
```

#### B3. Relax and extend remote validation

Two changes, matching the two-layer model from A3:

1. **Allow the new seed types.** Expand `_SUPPORTED_SEED_TYPES` in
   `unsupported_features.py:9` from `{"hf", "nmp"}` to also include
   `{"directory", "file_contents"}`. `agent_rollout` remains unsupported in
   remote mode.

2. **Extend pre-flight existence validation.** Extend `validate_seed`
   (`seed.py:20`) to handle `DirectorySeedSource` and
   `FileContentsSeedSource` the same way it already handles
   `FilesetFileSeedSource`: parse the `#`-ref via the existing
   `_parse_seed_source_path` (`seed.py:43`), then verify the fileset (and,
   where cheap, the path fragment) exists via `sdk.files.filesets.retrieve(...)`,
   surfacing `NDDInvalidConfigError` on 404 / `PermissionDeniedError`. This runs
   inside `RemoteDataDesignerContext.validate()` before workload submission, so
   the user gets fast feedback without a wasted job. Return the canonical
   `build_fileset_ref(...)` root on success; `RemoteDataDesignerContext.validate()`
   stores that value in `_validated_filesystem_roots`, and `get_seed_readers()`
   passes the set to `FilesetFileSystemProvider` so preview/job/SDK validation
   flows do not repeat the same Files existence check during engine compile or
   read setup.

### Server-side request validation (unchanged)

The existing `mode="before"` validator,
`validate_seed_source_for_execution_context` (`unsupported_features.py:38`),
stays as-is. It runs on the request models (`DataDesignerJobConfig` in
`jobs/spec.py:20`, `PreviewSpec` in `functions/_types.py:24`) and rejects
unsupported seed types by indexing into the raw request dict *before* Pydantic
hydrates the union member. We only expand `_SUPPORTED_SEED_TYPES` (B3) so it
admits the two new remote-capable types.

This validator must remain `mode="before"` because two source types cannot be
validated via normal typed hydration server-side:

- **`DataFrameSeedSource`** has a required `df: pd.DataFrame` field that is not
  JSON-serializable, so any `seed_type="df"` request arrives with `df` missing
  and fails Pydantic's "field required" before any typed validator could run.
- **`LocalFileSeedSource`** runs an `is_file()` existence check
  (`io_helpers.py:156`) against the *server's* disk during hydration, which the
  before-validator short-circuits by rejecting `local` as unsupported remotely
  (the client's file is not on the server, and `local` is not a remote type
  regardless).

Both share the property "cannot be meaningfully hydrated/validated as a typed
object server-side," so grouping them in a single raw-dict before-validator is
cohesive, not accidental.

> **Considered and dropped:** an earlier draft proposed deferring
> `LocalFileSeedSource`'s existence check (mirroring A3) so `local` would
> hydrate server-side, then splitting the rejection logic into a `df`-only
> before-validator plus a typed `mode="after"` validator. This was dropped: (1)
> `local` is unsupported in remote mode either way, so the work unlocks no new
> capability; (2) `df` still mandates a before-validator, so that code is not
> eliminated, only relocated — and splitting the unsupported-type logic across
> two validators in two modes is *less* cohesive than the single before-validator
> we have today. The corresponding upstream `LocalFileSeedSource` deferral is
> therefore also out of scope.

## Why this shape

- **Upstream stays generic.** It only learns about a `FileSystemProvider`
  protocol and an fsspec filesystem. No NeMo, no Filesets, no SDK references.
- **Dependency injection**, exactly the requested pattern: NeMo injects the
  provider via reader constructors — the same approach already used for
  `FilesetFileSeedReader(self._sdk)`.
- **Directory + FileContents are nearly free** because they already read via
  `context.fs`. The remaining work is provider injection plus validator
  relaxation.
- The existing `nmp` duckdb tabular path is untouched and remains the right
  tool for single/wildcard parquet reads.

## Alternatives considered

### Alt 1 — Stay NeMo-side only: expand `FilesetFileSeedReader`

Enhance the existing `nmp` reader/source to support wildcards and multi-file
globs (e.g. `*.parquet`) over duckdb, without touching upstream.

- **Pros:** No upstream changes; smallest blast radius.
- **Cons:** Only addresses tabular reads. Does not unlock the *semantics* of
  `directory` (file manifest rows) or `file_contents` (raw text per file).
  Users still can't do those remotely. Leaves the local/remote capability gap
  largely intact.

Rejected as the sole solution; may still be worth doing independently as a
quality-of-life improvement to the tabular path.

### Alt 2 — Make upstream seed sources fsspec-URL-aware

Teach the upstream sources to accept fsspec URLs (`fileset://...`,
`s3://...`) directly and resolve a filesystem from the URL protocol.

- **Pros:** Very generic; no provider injection.
- **Cons:** `FilesetFileSystem` needs an SDK instance, which cannot be encoded
  in a URL. We would still need a registration/injection mechanism for the SDK,
  reintroducing the same DI problem plus URL-protocol coupling. The provider
  seam is cleaner and keeps the SDK out of config strings.

### Alt 3 — Copy files to local disk before running (stage-in)

Download fileset contents to a temp dir in the service, then run the existing
local readers.

- **Pros:** Zero upstream changes.
- **Cons:** Doubles storage and I/O; complicates cleanup and large datasets;
  doesn't scale; worse than reading through fsspec directly.

## Risks and open questions

- **Agent rollout remains unsupported remotely.** This is intentional for this
  plan. If AgentRollout moves to a plugin, that plugin can own its own remote
  filesystem story without forcing handler abstractions into the core library.
- **Performance.** `directory` and `file_contents` perform per-file reads. Over
  Filesets each read is a network round trip. `FilesetFileSystem` is async with
  `batch_size=4`, but the hydration loop is currently sequential. Batched /
  concurrent hydration is future work, out of scope for the bridge itself.
- **Upstream change coordination.** Parts A1–A3 live in the DataDesigner repo
  and must land (and version-bump) before the NeMo wiring in B can depend on
  them.
- **Relative local path behavior change.** Local relative paths for `directory`
  and `file_contents` become run-context-relative rather than
  construction-cwd-relative. This is intentional for config portability, but it
  must be documented and covered by tests so future maintainers do not
  accidentally reintroduce hidden construction-cwd anchoring.
- **`root_path` typing.** Loosening `root_path` from `Path` to something more
  abstract may ripple through code that does `Path`-specific operations; an
  audit of all `context.root_path` uses is required. In the in-scope readers it
  should be metadata/rendering only.

## Implementation plan

Ordered to keep each step independently testable.

1. **Upstream A1 + A2** — add `FileSystemProvider` protocol, default local
   implementation, and loosen `root_path` so it can be a displayable remote
   root. Verify existing `directory` / `file_contents` upstream tests pass
   unchanged.
2. **NeMo B1 + partial B2** — add `FilesetFileSystemProvider`; wire
   `directory` + `file_contents` remotely. Integration-test against a real
   Fileset.
3. **A3 (Layer 1) — config refactor.** Drop `is_dir()` from the upstream
   `DirectorySeedSource` / `FileContentsSeedSource` path validation; make their
   `runtime_path` handling preserve the raw user-authored path string for the
   active provider to interpret. Keep `AgentRolloutSeedSource` local-path
   behavior unchanged. Add a **precise root-existence assertion** in
   `FileSystemSeedReader` / the provider so a missing root raises a clear "does
   not exist" error (local: via `LocalFileSystem`; fileset: via SDK) instead of
   the vague "no files matched". Verify upstream `DataDesigner.validate()` and
   the CLI still surface a precise error for a missing directory (now via the
   provider, reached through `get_column_names()`), and that existing tests pass.
   Add regression tests showing local relative paths are resolved at
   validate/read time and absolute paths remain stable across cwd changes.
4. **A3 (Layer 2, NeMo) + B3 — remote pre-flight validation.** Extend
   `validate_seed` to check `directory` / `file_contents` fileset refs via the
   SDK; expand `_SUPPORTED_SEED_TYPES`.
5. **SDK / OpenAPI** — regenerate only if request models change. They should
   not: the seed source config types already exist in the `dd` config and are
   serializable.

## Affected files (reference)

Upstream (`/Users/mknepper/code/DataDesigner`):

- `packages/data-designer-engine/src/data_designer/engine/resources/seed_reader.py`
  (A1, A2)
- `packages/data-designer-config/src/data_designer/config/seed_source.py`
  (A3 — `DirectorySeedSource` / `FileContentsSeedSource` deferral;
  `AgentRolloutSeedSource` and `LocalFileSeedSource` unchanged)

NeMo (`/Users/mknepper/code/nemo-platform`):

- `nemo-platform/packages/data_designer_nemo/src/data_designer_nemo/context.py` (B2)
- `nemo-platform/packages/data_designer_nemo/src/data_designer_nemo/` — new
  `fileset_filesystem_provider.py` (B1)
- `nemo-platform/packages/data_designer_nemo/src/data_designer_nemo/unsupported_features.py`
  (B3 — expand `_SUPPORTED_SEED_TYPES`)
- `nemo-platform/packages/data_designer_nemo/src/data_designer_nemo/seed.py` (A3 Layer 2 / B3)
