---
size_categories: {{ size_categories }}
tags:
{% for tag in tags %}
  - {{ tag }}
{% endfor %}
configs:
- config_name: data
  data_files: "data/*.parquet"
  default: true
{% if has_processors %}{% for processor_name in processor_names %}- config_name: {{ processor_name }}
  data_files: "{{ processor_name }}/*.parquet"
{% endfor %}{% endif %}
---

<div style="display: flex; justify-content: space-between; align-items: flex-end; width: 100%; margin-bottom: 1rem;">
<h1 style="flex: 1; margin: 0;">{{ repo_id.split('/')[-1] | title }}</h1>
<sub style="white-space: nowrap;">Made with ❤️ using 🎨 <a href="https://github.com/NVIDIA-NeMo/DataDesigner">NeMo Data Designer</a></sub>
</div>

---

{{ custom_description }}

---

## 🚀 Quick Start

```python
from datasets import load_dataset

# Load the main dataset
dataset = load_dataset("{{ repo_id }}", "data", split="train")
df = dataset.to_pandas()
{% if has_processors %}

# Load processor outputs (if available){% for processor_name in processor_names %}
processor_{{ processor_name }} = load_dataset("{{ repo_id }}", "{{ processor_name }}", split="train")
df_{{ processor_name }} = processor_{{ processor_name }}.to_pandas()
{% endfor %}{% endif %}
```

---

## 📊 Dataset Summary

- **📈 Records**: {{ "{:,}".format(num_records) }}
- **📋 Columns**: {{ num_columns }}
{% if target_num_records != num_records %}
- **✅ Completion**: {{ "%.1f" | format(percent_complete) }}% ({{ "{:,}".format(target_num_records) }} requested)
{% endif %}

---

## 📋 Schema & Statistics

{% if column_statistics %}
| Column | Type | Column Type | Unique (%) | Null (%) | Details |
|--------|------|-------------|------------|----------|---------|
{% for stat in column_statistics -%}
| `{{ stat.column_name }}` | `{{ stat.simple_dtype }}` | {{ stat.column_type }} | {{ stat.num_unique }} ({{ "%.1f" | format((stat.num_unique / stat.num_records * 100) if stat.num_records > 0 else 0) }}%) | {{ stat.num_null if stat.num_null > 0 else 0 }} ({{ "%.1f" | format((stat.num_null / stat.num_records * 100) if stat.num_records > 0 else 0) }}%) | {% if stat.column_type in ["llm-text", "llm-code", "llm-structured", "llm-judge"] %}Tokens: {{ "%.0f" | format(stat.output_tokens_mean) if stat.output_tokens_mean is defined else "N/A" }} out / {{ "%.0f" | format(stat.input_tokens_mean) if stat.input_tokens_mean is defined else "N/A" }} in{% elif stat.column_type == "sampler" and stat.sampler_type is defined %}{% if stat.sampler_type is mapping %}{{ stat.sampler_type.value }}{% else %}{{ stat.sampler_type }}{% endif %}{% else %}-{% endif %} |
{% endfor -%}
{% else %}
| Column | Type |
|--------|------|
{% for col_name, dtype in all_columns.items() | sort -%}
| `{{ col_name }}` | {{ dtype }} |
{% endfor %}
{% endif %}

---

## ⚙️ Generation Details

{% if config_types %}
Generated with {{ num_columns_configured }} column configuration(s):

{% for col_type, count in config_types.items() | sort %}
- **{{ col_type }}**: {{ count }} column(s)
{% endfor %}

{% endif %}
📄 Full configuration available in [`sdg.json`](sdg.json) and detailed metadata in [`metadata.json`](metadata.json).

---

## 📚 Citation

```bibtex
@misc{nemo-data-designer,
  author = {The NeMo Data Designer Team, NVIDIA},
  title = {NeMo Data Designer: A framework for generating synthetic data from scratch or based on your own seed data},
  howpublished = {\url{https://github.com/NVIDIA-NeMo/DataDesigner}},
  year = {{ current_year }},
  note = {GitHub Repository},
}
```

---

## 💡 About NeMo Data Designer

NeMo Data Designer is a general framework for generating high-quality synthetic data that goes beyond simple LLM prompting. It provides:

- **Diverse data generation** using statistical samplers, LLMs, or existing seed datasets
- **Relationship control** between fields with dependency-aware generation
- **Quality validation** with built-in Python, SQL, and custom local and remote validators
- **LLM-as-a-judge** scoring for quality assessment
- **Fast iteration** with preview mode before full-scale generation

For more information, visit: [https://github.com/NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) (`pip install data-designer`)
