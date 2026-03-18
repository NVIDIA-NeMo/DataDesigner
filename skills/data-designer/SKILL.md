---
name: data-designer
description: Use when the user wants to create a dataset, generate synthetic data, or build a data generation pipeline. Contains the essential workflow and discovery commands for the Data Designer library. Always invoke before exploring the workspace or writing code.
argument-hint: [describe the dataset you want to generate]
---

# Before You Start

Do not explore the workspace, browse files, run `ls`/`find`/`Glob`, check git history, or spawn Agent subagents before starting the workflow below.

# Goal

Build a synthetic dataset using the Data Designer library that matches this description:

$ARGUMENTS

# Agent CLI

Always run this command before attempting to design or build a dataset. This command is your single discovery mechanism — it tells you exactly which files to read:

```bash
data-designer agent context
```

# Workflow

Use **Autopilot** mode if the user implies they don't want to answer questions — e.g., they say something like "be opinionated", "you decide", "make reasonable assumptions", "just build it", "surprise me", etc. Otherwise, use **Interactive** mode (default).

Read **only** the workflow file that matches the selected mode, then follow it:

- **Interactive** → read `workflows/interactive.md`
- **Autopilot** → read `workflows/autopilot.md`

# Rules

- Do not drop columns unless the user explicitly asks. Keep all columns in the output by default.
- Do not suggest or ask about seed datasets. Only use one when the user explicitly provides seed data or asks to build from existing records. When using a seed, read `references/seed-datasets.md`.
- When the dataset requires person data (names, demographics, addresses), read `references/person-sampling.md`.
- If a dataset script that matches the dataset description already exists, ask the user whether to edit it or create a new one.

# Usage Tips and Common Pitfalls

- **Sampler and validation columns need both a type and params.** E.g., `sampler_type="category"` with `params=dd.CategorySamplerParams(...)`.
- **Jinja2 templates** in `prompt`, `system_prompt`, and `expr` fields: reference columns with `{{ column_name }}`, nested fields with `{{ column_name.field }}`.
- **`SamplerColumnConfig`:** Takes `params`, not `sampler_params`.
- **LLM judge score access:** `LLMJudgeColumnConfig` produces a nested dict where each score name maps to `{reasoning: str, score: int}`. To get the numeric score, use the `.score` attribute. For example, for a judge column named `quality` with a score named `correctness`, use `{{ quality.correctness.score }}`. Using `{{ quality.correctness }}` returns the full dict, not the numeric score.
- **Nested field access in `SchemaTransformProcessorConfig`:** Nested field access (e.g., `{{ column.field }}`) does **not** work inside schema transform templates because the processor sees column values as serialized strings, not parsed dicts. This affects structured columns, judge columns, and any column with nested output. To use nested fields in a schema transform, first extract them into intermediate `ExpressionColumnConfig` columns (e.g., `expr="{{ column.field }}"` with `drop=True`), then reference those flat columns in the template.

# Troubleshooting

- **`data-designer` command not found:** If no virtual environment exists, create one first (`python -m venv .venv && source .venv/bin/activate`), then install (`pip install data-designer`). If a virtual environment already exists, activate it and verify the package is installed.
- **Network errors during preview:** A sandbox environment may be blocking outbound requests. Ask the user for permission to retry the command with the sandbox disabled. Only as a last resort, if retrying outside the sandbox also fails, tell the user to run the command themselves.

# Output Template

Write a Python file to the current directory with a `load_config_builder()` function returning a `DataDesignerConfigBuilder`. Use PEP 723 inline metadata for dependencies.

```python
# /// script
# dependencies = [
#   "data-designer",
#   "pydantic",
# ]
# ///
import data_designer.config as dd
from pydantic import BaseModel, Field


# Define Pydantic models when a column needs structured output
class MyEntity(BaseModel):
    field_one: str = Field(description="...")
    field_two: int = Field(description="...")


# Use custom generators when built-in column types aren't enough
@dd.custom_column_generator(
    required_columns=["col_a"],
    side_effect_columns=["extra_col"],
)
def my_custom_generator(row: dict) -> dict:
    # add custom logic here that depends on "col_a" and update row in place
    row["name_in_custom_column_config"] = "custom value"
    row["extra_col"] = "extra value"
    return row


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    # Seed dataset (only if the user explicitly mentions a seed dataset path)
    # config_builder.with_seed_dataset(dd.LocalFileSeedSource(path="path/to/seed.parquet"))

    # config_builder.add_column(...)
    # config_builder.add_processor(...)

    return config_builder
```

Only include Pydantic models, custom generators, seed datasets, and extra dependencies when the task requires them.
