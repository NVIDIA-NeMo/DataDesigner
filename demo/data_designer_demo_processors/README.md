# Demo Processor Plugins for Data Designer

Two example processor plugins demonstrating the Data Designer plugin system.

## Processors

### RegexFilterProcessor (`regex-filter`)

Filters rows by regex pattern on a specified column. Runs at `process_before_batch`.

```python
config_builder.add_processor(
    RegexFilterProcessorConfig(
        name="filter_english",
        column="language",
        pattern="^(en|english)$",
        invert=False,
    )
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `column` | `str` | required | Column to match against |
| `pattern` | `str` | required | Regex pattern |
| `invert` | `bool` | `False` | Keep non-matching rows instead |

### SemanticDedupProcessor (`semantic-dedup`)

Removes near-duplicate rows using embedding cosine similarity. Runs at `process_after_generation`.

```python
config_builder.add_processor(
    SemanticDedupProcessorConfig(
        name="dedup",
        column="generated_text",
        similarity_threshold=0.9,
    )
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `column` | `str` | required | Column to compute embeddings on |
| `similarity_threshold` | `float` | `0.9` | Cosine similarity threshold |
| `model_name` | `str` | `all-MiniLM-L6-v2` | Sentence-transformers model |

## Installation

```bash
uv pip install -e demo/data_designer_demo_processors
```

## Entry Points

```toml
[project.entry-points."data_designer.plugins"]
regex-filter = "data_designer_demo_processors.regex_filter.plugin:regex_filter_plugin"
semantic-dedup = "data_designer_demo_processors.semantic_dedup.plugin:semantic_dedup_plugin"
```
