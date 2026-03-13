---
date: 2026-03-13
authors:
  - etramel
---

# **Handler Registries for Lightweight Runtime Extensions**

Data Designer already has a strong abstraction for heavyweight engine tasks: `ConfigurableTask` plus `TaskRegistry`. That pairing works well for column generators, processors, and profilers because those components participate in the full engine lifecycle and need resources such as artifact storage and model access.

Some runtime extension points are much lighter than that. Seed readers are the clearest example in `main` today. A seed reader is just a config-selected handler that knows how to attach a seed source and expose it to DuckDB. It does not need the full `ConfigurableTask` contract, but it still benefits from a consistent registry pattern.

The new `HandlerRegistry` fills that gap. It is an instance-scoped registry for lightweight handlers selected by a config discriminator. The registry takes three inputs:

1. The config discriminator field, such as `seed_type`.
2. A lightweight handler base class that declares its config type through generics.
3. A factory for constructing the handler once the matching handler class has been resolved.

That lets us use the same pattern for seed readers today and for future lightweight runtime strategies without forcing them into `TaskRegistry`.

## **What It Looks Like for Seed Readers**

The built-in seed readers now register by class instead of by shared singleton-like instances:

```python
registry = SeedReaderRegistry(
    readers=[
        HuggingFaceSeedReader,
        LocalFileSeedReader,
        DataFrameSeedReader,
    ]
)
```

At lookup time, the registry reads `seed_type` from the `SeedSource`, resolves the handler class, instantiates a fresh reader, and then attaches the source and secret resolver.

## **Future Example: `GitRepoSeedSource`**

This PR does not add a git repository seed source, but it is a good example of the intended extension shape:

```python
class GitRepoSeedSource(SeedSource):
    seed_type: Literal["git-repo"] = "git-repo"
    repo_url: str
    ref: str = "main"


class GitRepoSeedReader(SeedReader[GitRepoSeedSource]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        ...

    def get_dataset_uri(self) -> str:
        ...
```

Booting that feature up would be straightforward:

```python
registry = SeedReaderRegistry(
    readers=[
        HuggingFaceSeedReader,
        LocalFileSeedReader,
        DataFrameSeedReader,
        GitRepoSeedReader,
    ]
)
```

The important point is that `GitRepoSeedReader` is still a lightweight handler. It is selected from `seed_type`, but it does not need to become a `ConfigurableTask` just to participate in registry lookup.

## **Why This Matters**

This gives us a middle-sized abstraction between:

- `TaskRegistry` for full engine tasks
- one-off ad hoc registries for lightweight runtime handlers

That keeps the heavy abstractions focused on heavy components while giving lighter runtime extension points a consistent, reusable pattern.
