# Data Designer Demo Processors

Demo processor plugins demonstrating PRE_GENERATION and POST_GENERATION stages.

## Installation

```bash
uv pip install -e demo/data_designer_demo_processors
```

## Processors

### RegexFilterProcessor (PRE_GENERATION)

Filters seed data rows based on regex pattern matching.

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer_demo_processors.regex_filter import RegexFilterProcessorConfig

builder = DataDesignerConfigBuilder(model_configs=[...])
builder.add_processor(RegexFilterProcessorConfig(
    name="filter_emails",
    column="email",
    pattern=r"@company\.com$",
    invert=False,  # Keep only matching rows
))
```

### SemanticDedupProcessor (POST_GENERATION)

Removes semantically similar rows using sentence embeddings.

```python
from data_designer_demo_processors.semantic_dedup import SemanticDedupProcessorConfig

builder.add_processor(SemanticDedupProcessorConfig(
    name="dedup_responses",
    column="response",
    similarity_threshold=0.9,  # Remove rows with >90% similarity
    model_name="all-MiniLM-L6-v2",
))
```

## Pre-downloading the Embedding Model

The semantic dedup processor downloads the embedding model on first use. To pre-download:

```bash
download-semantic-dedup-model
```

## Entry Points

The package registers plugins via entry points:

```toml
[project.entry-points."data_designer.plugins"]
regex-filter = "data_designer_demo_processors.regex_filter.plugin:regex_filter_plugin"
semantic-dedup = "data_designer_demo_processors.semantic_dedup.plugin:semantic_dedup_plugin"
```
