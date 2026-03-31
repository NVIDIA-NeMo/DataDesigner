# FileSystemSeedReader Plugins

!!! warning "Experimental Feature"
    The plugin system is currently **experimental** and under active development. The documentation, examples, and plugin interface are subject to significant changes in future releases. If you encounter any issues, have questions, or have ideas for improvement, please consider starting [a discussion on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/discussions).

`FileSystemSeedReader` is the simplest way to build a seed reader plugin when your source data lives in a directory of files. You describe the files cheaply in `build_manifest(...)`, then optionally read and reshape them in `hydrate_row(...)`.

This guide focuses on the filesystem-specific contract. The fastest way to learn it is usually to start with an inline reader over `DirectorySeedSource`, then package that reader later only if you need automatic plugin discovery or a brand-new `seed_type`. For a runnable single-file example, see the [Markdown Section Seed Reader recipe](../recipes/plugin_development/markdown_seed_reader.md).

## What the framework owns

When you inherit from `FileSystemSeedReader`, Data Designer already handles:

- attachment-scoped filesystem context reuse
- file matching with `file_pattern` and `recursive`
- manifest sampling, `IndexRange`, `PartitionBlock`, and shuffle
- batching and DuckDB registration
- hydrated output schema validation via `output_columns`

Most readers only need to implement `build_manifest(...)` and `hydrate_row(...)`.

## Start with an existing filesystem config

If your source data already fits `DirectorySeedSource` or `FileContentsSeedSource`, you do not need a new config model just to learn or prototype a reader. Reuse the built-in source type and override how one `DataDesigner` instance interprets that seed source.

The Markdown recipe uses `DirectorySeedSource(path=..., file_pattern="*.md")` and pairs it with an inline reader:

```python
import data_designer.config as dd
from pathlib import Path
from typing import Any

from data_designer.engine.resources.seed_reader import FileSystemSeedReader, SeedReaderFileSystemContext


class MarkdownSectionDirectorySeedReader(FileSystemSeedReader[dd.DirectorySeedSource]):
    output_columns = [
        "relative_path",
        "file_name",
        "section_index",
        "section_header",
        "section_content",
    ]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                "file_name": Path(relative_path).name,
            }
            for relative_path in matched_paths
        ]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        ...
```

This approach lets you inspect the manifest and hydration contract without first creating a package, entry points, or a new `seed_type`.

## Step 1: Build a cheap manifest

`build_manifest(...)` should be inexpensive. Usually that means enumerating matching files and returning one logical row per file, without reading file contents yet.

In this example, the manifest only tracks:

- `relative_path`
- `file_name`

That keeps selection and partitioning file-based.

## Step 2: Hydrate one file into one or many rows

`hydrate_row(...)` can return either:

- a single record dict for `1:1` hydration
- an iterable of record dicts for `1:N` hydration

If hydration changes the schema, set `output_columns` to the exact emitted schema:

```python
output_columns = [
    "relative_path",
    "file_name",
    "section_index",
    "section_header",
    "section_content",
]
```

In the recipe implementation, `hydrate_row(...)` reads one file and emits one record per ATX heading section.

Every emitted record must match `output_columns` exactly. Data Designer will raise a plugin-facing error if a hydrated record is missing a declared column or includes an undeclared one.

## Step 3: Pass the reader to Data Designer

Register the inline reader on the `DataDesigner` instance you want to use:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner(seed_readers=[MarkdownSectionDirectorySeedReader()])

builder = dd.DataDesignerConfigBuilder()
builder.with_seed_dataset(
    dd.DirectorySeedSource(path="sample_data", file_pattern="*.md"),
)
```

That pattern overrides how this `DataDesigner` instance handles the built-in `directory` seed source. Because `seed_readers` sets the registry for that instance, include any other readers you still want available. This is a good fit for local experiments, tests, and docs recipes.

## Manifest-Based Selection Semantics

Selection stays manifest-based even when `hydrate_row(...)` fans out.

If the matched files are:

```text
0 -> faq.md
1 -> guide.md
```

and `guide.md` hydrates into two section rows, then:

```python
import data_designer.config as dd
from data_designer.config.seed import IndexRange

builder.with_seed_dataset(
    dd.DirectorySeedSource(path="sample_data", file_pattern="*.md"),
    selection_strategy=IndexRange(start=1, end=1),
)
```

selects only `guide.md`, then returns **all** section rows emitted from `guide.md`.

That means `get_seed_dataset_size()`, `IndexRange`, `PartitionBlock`, and shuffle all operate on manifest rows before hydration.

## Package it later when needed

If you want the same reader to be installable and auto-discovered as a plugin, then move from the inline pattern to a package:

- define a config class that inherits from `FileSystemSeedSource`
- give it a unique `seed_type`
- create a `Plugin` object with `plugin_type=PluginType.SEED_READER`
- register that plugin via a `data_designer.plugins` entry point

That extra packaging step is only necessary when you need a reusable plugin boundary. The reader logic itself still lives in the same `build_manifest(...)` and `hydrate_row(...)` methods shown above.

## Advanced Hooks

If you need more control, `FileSystemSeedReader` also lets you override:

- `on_attach(...)` for per-attachment setup
- `create_filesystem_context(...)` for custom rooted filesystem behavior

Most filesystem plugins do not need either hook.
