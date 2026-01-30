---
library: datadesigner
size_categories: {{ size_categories }}
tags:
  - synthetic
  - nemo-data-designer
---

# {{ repo_id.split('/')[-1] | title }}

This dataset was generated using **[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)**, a framework for creating high-quality synthetic datasets.

## Dataset Summary

- **Records**: {{ "{:,}".format(num_records) }}
- **Columns**: {{ num_columns }}
{% if target_num_records != num_records %}
- **Completion**: {{ "%.1f" | format(percent_complete) }}% ({{ "{:,}".format(target_num_records) }} requested)
{% endif %}

## Quick Start

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{{ repo_id }}")
df = dataset["train"].to_pandas()
```

## Schema & Statistics

{% if column_statistics %}
{% for stat in column_statistics %}
### {{ stat.column_name }}

- **Type**: `{{ stat.simple_dtype }}`
- **Column Type**: {{ stat.column_type }}
- **Unique Values**: {{ stat.num_unique }} ({{ "%.1f" | format((stat.num_unique / stat.num_records * 100) if stat.num_records > 0 else 0) }}%)
{% if stat.num_null > 0 %}
- **Null Values**: {{ stat.num_null }} ({{ "%.1f" | format((stat.num_null / stat.num_records * 100) if stat.num_records > 0 else 0) }}%)
{% endif %}
{% if stat.column_type in ["llm-text", "llm-code", "llm-structured", "llm-judge"] %}
- **Avg Output Tokens**: {{ "%.1f" | format(stat.output_tokens_mean) if stat.output_tokens_mean is defined else "N/A" }}
- **Avg Input Tokens**: {{ "%.1f" | format(stat.input_tokens_mean) if stat.input_tokens_mean is defined else "N/A" }}
{% endif %}
{% if stat.column_type == "sampler" and stat.sampler_type is defined %}
- **Sampler Type**: {% if stat.sampler_type is mapping %}{{ stat.sampler_type.value }}{% else %}{{ stat.sampler_type }}{% endif %}
{% endif %}

{% endfor %}
{% else %}
| Column | Type |
|--------|------|
{% for col_name, dtype in all_columns.items() | sort -%}
| `{{ col_name }}` | {{ dtype }} |
{% endfor %}
{% endif %}

## Generation Details

{% if config_types %}
Generated with {{ num_columns_configured }} column configuration(s):

{% for col_type, count in config_types.items() | sort %}
- **{{ col_type }}**: {{ count }} column(s)
{% endfor %}
{% endif %}

Full configuration available in `sdg.json` and detailed metadata in `metadata.json`.

## Citation

```bibtex
@misc{nemo-data-designer,
  author = {The NeMo Data Designer Team, NVIDIA},
  title = {NeMo Data Designer: A framework for generating synthetic data from scratch or based on your own seed data},
  howpublished = {\url{https://github.com/NVIDIA-NeMo/DataDesigner}},
  year = {{ current_year }},
  note = {GitHub Repository},
}
```
