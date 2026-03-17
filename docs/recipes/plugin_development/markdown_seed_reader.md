# Markdown Section Seed Reader Plugin

Turn a directory of Markdown files into a seed dataset with one row per section. This recipe is a realistic multi-file plugin scaffold built around `FileSystemSeedReader` fanout in `hydrate_row(...)`.

## What it demonstrates

- authoring a `FileSystemSeedReader` plugin package
- returning `1:N` hydrated rows from one manifest row
- declaring `output_columns` for the hydrated schema
- keeping `IndexRange` selection manifest-based
- installing a plugin locally with `uv pip install -e .`

## Scaffold Layout

```text
markdown_seed_reader/
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

## Run the Recipe

1. Recreate this directory layout locally.
2. From the scaffold root, run `uv pip install -e .`
3. Run `uv run demo.py`

`demo.py` prints two previews:

- the full section dataset across all Markdown files
- a manifest-only selection using `IndexRange(start=1, end=1)` that still returns every section from the selected file

## pyproject.toml

[Download `pyproject.toml` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/pyproject.toml){ .md-button download="pyproject.toml" }

```toml
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/pyproject.toml"
```

## src/data_designer_markdown_seed_reader/__init__.py

[Download `__init__.py` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/__init__.py){ .md-button download="__init__.py" }

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/__init__.py"
```

## src/data_designer_markdown_seed_reader/config.py

[Download `config.py` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/config.py){ .md-button download="config.py" }

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/config.py"
```

## src/data_designer_markdown_seed_reader/impl.py

[Download `impl.py` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/impl.py){ .md-button download="impl.py" }

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/impl.py"
```

## src/data_designer_markdown_seed_reader/plugin.py

[Download `plugin.py` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/plugin.py){ .md-button download="plugin.py" }

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/src/data_designer_markdown_seed_reader/plugin.py"
```

## demo.py

[Download `demo.py` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/demo.py){ .md-button download="demo.py" }

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/demo.py"
```

## sample_data/faq.md

[Download `faq.md` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/sample_data/faq.md){ .md-button download="faq.md" }

```markdown
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/sample_data/faq.md"
```

## sample_data/guide.md

[Download `guide.md` :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader/sample_data/guide.md){ .md-button download="guide.md" }

```markdown
--8<-- "assets/recipes/plugin_development/markdown_seed_reader/sample_data/guide.md"
```
