# Markdown Section Seed Reader

Turn a directory of Markdown files into a seed dataset with one row per section. This recipe stays in the same single-file format as the other recipes: it creates sample files, defines an inline `FileSystemSeedReader[DirectorySeedSource]`, and passes that reader to `DataDesigner(seed_readers=[...])`.

This keeps the example focused on the actual seed reader contract:

- implementing `build_manifest(...)`
- returning `1:N` hydrated rows from `hydrate_row(...)`
- declaring `output_columns` for the hydrated schema
- keeping `IndexRange` selection manifest-based

Because the example reuses `DirectorySeedSource`, it does not register a brand-new `seed_type`. If you later want to package the same reader as an installable plugin, see [FileSystemSeedReader Plugins](../../plugins/filesystem_seed_reader.md).

## Run the Recipe

Run the script directly:

```bash
uv run markdown_seed_reader.py
```

The script prints two previews:

- the full section dataset across all Markdown files
- a manifest-only selection using `IndexRange(start=1, end=1)` that still returns every section from the selected file

[Download Code :octicons-download-24:](../../assets/recipes/plugin_development/markdown_seed_reader.py){ .md-button download="markdown_seed_reader.py" }

```python
--8<-- "assets/recipes/plugin_development/markdown_seed_reader.py"
```
