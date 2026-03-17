# FileSystemSeedReader Plugins

!!! warning "Experimental Feature"
    The plugin system is currently **experimental** and under active development. The documentation, examples, and plugin interface are subject to significant changes in future releases. If you encounter any issues, have questions, or have ideas for improvement, please consider starting [a discussion on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/discussions).

`FileSystemSeedReader` is the simplest way to build a seed reader plugin when your source data lives in a directory of files. You describe the files cheaply in `build_manifest(...)`, then optionally read and reshape them in `hydrate_row(...)`.

This guide focuses on the filesystem-specific contract. For a runnable end-to-end scaffold, see the [Markdown Section Seed Reader recipe](../recipes/plugin_development/markdown_seed_reader.md).

## What the framework owns

When you inherit from `FileSystemSeedReader`, Data Designer already handles:

- attachment-scoped filesystem context reuse
- file matching with `file_pattern` and `recursive`
- manifest sampling, `IndexRange`, `PartitionBlock`, and shuffle
- batching and DuckDB registration
- hydrated output schema validation via `output_columns`

Most plugins only need to implement `build_manifest(...)` and `hydrate_row(...)`.

## Recommended package layout

```text
data-designer-markdown-seed-reader/
├── pyproject.toml
├── demo.py
├── sample_data/
│   ├── faq.md
│   └── guide.md
└── src/
    └── data_designer_markdown_seed_reader/
        ├── __init__.py
        ├── config.py
        ├── impl.py
        └── plugin.py
```

## Step 1: Define the seed source config

Your config class should inherit from `FileSystemSeedSource` and declare a unique `seed_type`.

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/config.py"
```

**Key points:**

- `path` points at a directory, not an individual file
- `file_pattern` matches basenames only
- `seed_type` is the discriminator and plugin identity for the reader

## Step 2: Build a cheap manifest

`build_manifest(...)` should be inexpensive. Usually that means enumerating matching files and returning one logical row per file, without reading file contents yet.

The Markdown example keeps the manifest file-based and performs the expensive parsing in `hydrate_row(...)`:

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/impl.py"
```

In this example, the manifest only tracks:

- `relative_path`
- `file_name`

That keeps selection and partitioning file-based.

## Step 3: Hydrate one file into one or many rows

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

In that implementation, `hydrate_row(...)` reads one file and emits one record per ATX heading section.

Every emitted record must match `output_columns` exactly. Data Designer will raise a plugin-facing error if a hydrated record is missing a declared column or includes an undeclared one.

## Step 4: Register the plugin

Create a `Plugin` object and register it as a `SEED_READER` entry point:

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/plugin.py"
```

```toml
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/pyproject.toml"
```

Install the package locally:

```bash
uv pip install -e .
```

## Manifest-Based Selection Semantics

Selection stays manifest-based even when `hydrate_row(...)` fans out.

If the matched files are:

```text
0 -> faq.md
1 -> guide.md
```

and `guide.md` hydrates into two section rows, then:

```python
from data_designer.config.seed import IndexRange

builder.with_seed_dataset(
    MarkdownSectionSeedSource(path="sample_data"),
    selection_strategy=IndexRange(start=1, end=1),
)
```

selects only `guide.md`, then returns **all** section rows emitted from `guide.md`.

That means `get_seed_dataset_size()`, `IndexRange`, `PartitionBlock`, and shuffle all operate on manifest rows before hydration.

## Validate the Plugin

Use `assert_valid_plugin` while developing the package:

```python
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer_markdown_seed_reader.plugin import markdown_section_seed_reader_plugin

assert_valid_plugin(markdown_section_seed_reader_plugin)
```

## Advanced Hooks

If you need more control, `FileSystemSeedReader` also lets you override:

- `on_attach(...)` for per-attachment setup
- `create_filesystem_context(...)` for custom rooted filesystem behavior

Most filesystem plugins do not need either hook.
