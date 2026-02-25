"""Programming Task Dataset — structured output + code generation patterns.

PATTERN REFERENCE ONLY — copy the structure, not the domain-specific values.

Demonstrates:
- Default model aliases (nvidia-text) — no manual ModelConfig needed
- Pydantic BaseModel as output_format for LLMStructuredColumnConfig
- Nested field access in downstream prompts ({{ task_spec.function_name }})
- LLMCodeColumnConfig with CodeLang.PYTHON
- CATEGORY sampler (uniform, no weights)
- preview() + create() workflow
"""

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()

config_builder = dd.DataDesignerConfigBuilder()

# --- Sampler columns ---

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="domain",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "data processing",
                "string manipulation",
                "math",
                "file I/O",
                "API calls",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="difficulty",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["easy", "medium", "hard"]),
    )
)

# --- Pydantic schema for structured output ---


class TaskSpec(BaseModel):
    task_description: str = Field(description="What the function should do")
    function_name: str = Field(description="Name of the function")
    input_params: list[str] = Field(description="List of parameter names")
    return_type: str = Field(description="Expected return type")


# --- Structured LLM column ---

config_builder.add_column(
    dd.LLMStructuredColumnConfig(
        name="task_spec",
        prompt=(
            "Create a {{ difficulty }} programming task in the {{ domain }} domain. "
            "Define a function specification with a clear description, name, "
            "parameters, and return type."
        ),
        output_format=TaskSpec,
        model_alias="nvidia-text",
    )
)

# --- Code generation column (references nested structured fields) ---

config_builder.add_column(
    dd.LLMCodeColumnConfig(
        name="solution",
        prompt=(
            "Write a Python function based on this specification:\n\n"
            "Task: {{ task_spec.task_description }}\n"
            "Function name: {{ task_spec.function_name }}\n"
            "Parameters: {{ task_spec.input_params }}\n"
            "Return type: {{ task_spec.return_type }}\n\n"
            "Include docstring and type hints."
        ),
        code_lang=dd.CodeLang.PYTHON,
        model_alias="nvidia-text",
    )
)

# --- Preview then create ---

preview = data_designer.preview(config_builder, num_records=3)
preview.display_sample_record()

results = data_designer.create(config_builder, num_records=100, dataset_name="programming-tasks")
dataset = results.load_dataset()
analysis = results.load_analysis()
