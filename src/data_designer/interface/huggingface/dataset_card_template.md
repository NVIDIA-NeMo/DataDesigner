---
size_categories: {{ size_categories }}
tags:
{% for tag in tags %}
  - {{ tag }}
{% endfor %}
---

# Dataset Card

This dataset was generated using **NeMo Data Designer**, a comprehensive framework for creating high-quality synthetic datasets from scratch or using seed data.

## About NeMo Data Designer

NeMo Data Designer is a general framework for generating high-quality synthetic data that goes beyond simple LLM prompting. It provides:

- **Diverse data generation** using statistical samplers, LLMs, or existing seed datasets
- **Relationship control** between fields with dependency-aware generation
- **Quality validation** with built-in Python, SQL, and custom local and remote validators
- **LLM-as-a-judge** scoring for quality assessment
- **Fast iteration** with preview mode before full-scale generation

For more information, visit: [https://github.com/NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner)

## Quick Start

Load this dataset for fine-tuning:

```python
from datasets import load_dataset

dataset = load_dataset("{{ repo_id }}")
# Access the data
df = dataset["train"].to_pandas()
```

Or with NeMo Data Designer:

```python
from data_designer.interface.results import DatasetCreationResults

# Load dataset with all artifacts (analysis, configs, etc.)
results = DatasetCreationResults.pull_from_hub("{{ repo_id }}")

# Access the dataset
df = results.load_dataset()

# Access the analysis
analysis = results.load_analysis()

# Access the config builder
config_builder = results._config_builder
```

## Dataset Summary

- **Number of records**: {% if num_records is defined and num_records is not none %}{{ "{:,}".format(num_records) }}{% else %}N/A{% endif %}
- **Number of columns**: {{ num_columns }}
- **Size category**: {{ size_categories }}
{% if target_num_records is defined and target_num_records is not none and target_num_records != num_records %}
- **Target records**: {{ "{:,}".format(target_num_records) }} ({{ "%.1f" | format(percent_complete) if percent_complete is defined and percent_complete is not none else "N/A" }}% complete)
{% endif %}

## Sample Data

{% if num_samples > 0 %}
Here are sample records from the dataset:

{% for idx in range(num_samples) %}
### Example {{ idx + 1 }}

```json
{{ sample_records[idx] | tojson(indent=2) }}
```
{% endfor %}
{% else %}
No sample records available.
{% endif %}

## Schema

{% if all_columns is defined and all_columns %}
| Column | Type | Description |
|--------|------|-------------|
{% for col_name, dtype in all_columns | dictsort -%}
| `{{ col_name }}` | {{ dtype }} | {% if column_configs %}{% for col_config in column_configs %}{% if col_config.get('name') == col_name %}{% set col_type = col_config.get('column_type') %}{% if col_type is mapping %}{{ col_type.get('value', '') }}{% elif col_type %}{{ col_type }}{% endif %}{% endif %}{% endfor %}{% endif %} |
{% endfor -%}
{% else %}
No column information available.
{% endif %}

## Data Quality

{% if column_stats_by_type %}
### Column Statistics

{% for col_type in sorted_column_types %}
{% set stats_list = column_stats_by_type[col_type] %}
{% if stats_list %}
{% set col_type_label = col_type.replace("_", " ").title().replace("Llm", "LLM") %}
#### {{ col_type_label }} Columns

{% if col_type == "sampler" %}
| Column | Data Type | Unique Values | Sampler Type |
|--------|-----------|---------------|--------------|
{% for stat in stats_list -%}
| **{{ stat.get('column_name', 'unknown') }}** | {{ stat.get('simple_dtype', 'unknown') }} | {% if 'num_unique' in stat and stat['num_unique'] is not none %}{{ stat['num_unique'] }}{% else %}N/A{% endif %} ({% if 'num_unique' in stat and stat['num_unique'] is not none and num_records > 0 %}{{ "%.1f" | format((stat['num_unique'] / num_records * 100)) }}{% else %}0.0{% endif %}%) | {% if 'sampler_type' in stat and stat['sampler_type'] is not none %}{% set sampler_type = stat['sampler_type'] %}{% if sampler_type is mapping %}{{ sampler_type.get('value', 'N/A') }}{% else %}{{ sampler_type }}{% endif %}{% else %}N/A{% endif %} |
{% endfor -%}

{% elif col_type in ["llm_text", "llm_structured", "llm_code", "llm_judge"] %}
| Column | Data Type | Unique Values | Prompt Tokens (avg) | Completion Tokens (avg) |
|--------|-----------|---------------|---------------------|--------------------------|
{% for stat in stats_list -%}
| **{{ stat.get('column_name', 'unknown') }}** | {{ stat.get('simple_dtype', 'unknown') }} | {% if 'num_unique' in stat and stat['num_unique'] is not none %}{{ stat['num_unique'] }}{% else %}N/A{% endif %} ({% if 'num_unique' in stat and stat['num_unique'] is not none and num_records > 0 %}{{ "%.1f" | format((stat['num_unique'] / num_records * 100)) }}{% else %}0.0{% endif %}%) | {% if 'prompt_tokens_mean' in stat and stat['prompt_tokens_mean'] is not none %}{{ "%.1f" | format(stat['prompt_tokens_mean']) }}{% else %}N/A{% endif %} ± {% if 'prompt_tokens_stddev' in stat and stat['prompt_tokens_stddev'] is not none %}{{ "%.1f" | format(stat['prompt_tokens_stddev']) }}{% else %}N/A{% endif %} | {% if 'completion_tokens_mean' in stat and stat['completion_tokens_mean'] is not none %}{{ "%.1f" | format(stat['completion_tokens_mean']) }}{% else %}N/A{% endif %} ± {% if 'completion_tokens_stddev' in stat and stat['completion_tokens_stddev'] is not none %}{{ "%.1f" | format(stat['completion_tokens_stddev']) }}{% else %}N/A{% endif %} |
{% endfor -%}

{% else %}
| Column | Data Type | Unique Values | Null Values |
|--------|-----------|---------------|-------------|
{% for stat in stats_list -%}
| **{{ stat.get('column_name', 'unknown') }}** | {{ stat.get('simple_dtype', 'unknown') }} | {% if 'num_unique' in stat and stat['num_unique'] is not none %}{{ stat['num_unique'] }}{% else %}N/A{% endif %} ({% if 'num_unique' in stat and stat['num_unique'] is not none and num_records > 0 %}{{ "%.1f" | format((stat['num_unique'] / num_records * 100)) }}{% else %}0.0{% endif %}%) | {% if 'num_null' in stat and stat['num_null'] is not none %}{{ stat['num_null'] }}{% else %}0{% endif %} ({% if 'num_null' in stat and stat['num_null'] is not none and num_records > 0 %}{{ "%.1f" | format((stat['num_null'] / num_records * 100)) }}{% else %}0.0{% endif %}%) |
{% endfor -%}
{% endif %}
{% endif %}

{% endfor %}
{% elif column_statistics %}
{% for stat in column_statistics[:10] %}
- **{{ stat.get('column_name', 'unknown') }}** ({{ stat.get('column_type', 'unknown') }}): {% if 'num_unique' in stat and stat['num_unique'] is not none %}{{ stat['num_unique'] }} unique values{% if num_records > 0 %} ({{ "%.1f" | format((stat['num_unique'] / num_records * 100)) }}% coverage){% endif %}{% else %}N/A{% endif %}{% if 'num_null' in stat and stat['num_null'] is not none and stat['num_null'] > 0 %}, {{ stat['num_null'] }} nulls{% endif %}
{% endfor %}
{% if column_statistics | length > 10 %}
*... and {{ (column_statistics | length) - 10 }} more columns*
{% endif %}
{% endif %}

## Configuration Details

{% if column_configs %}
This dataset was generated with {{ column_configs | length }} column configuration(s).

### Generation Strategy

{% for config_type, count in config_types | dictsort %}
- **{{ config_type }}**: {{ count }} column(s)
{% endfor %}

### Column Configurations

{% for col_config in column_configs %}
- **{{ col_config.get('name', 'unknown') }}**: {% set col_type = col_config.get('column_type') %}{% if col_type is mapping %}{{ col_type.get('value', 'unknown') }}{% elif col_type %}{{ col_type }}{% else %}unknown{% endif %}
{% endfor %}
{% else %}
No column configurations available.
{% endif %}

{% if metadata %}
## Metadata

```json
{{ metadata | tojson(indent=2) }}
```
{% endif %}

## Citation

If you use this dataset in your research, please cite:

```bibtex
@software{data_designer,
  title={NeMo Data Designer: A Framework for Synthetic Dataset Generation},
  author={NVIDIA},
  year={2025},
  url={https://github.com/NVIDIA-NeMo/DataDesigner}
}
```
