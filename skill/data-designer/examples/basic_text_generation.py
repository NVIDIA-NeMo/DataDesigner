"""Product Review Dataset — basic text generation patterns.

PATTERN REFERENCE ONLY — copy the structure, not the domain-specific values.

Demonstrates:
- Default model aliases (nvidia-text) — no manual ModelConfig needed
- CATEGORY sampler with weights
- PERSON_FROM_FAKER sampler with drop=True
- UNIFORM sampler with convert_to="int"
- ExpressionColumnConfig for derived fields
- LLMTextColumnConfig with Jinja2 templates referencing nested fields
- preview() + create() workflow
"""

import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()

config_builder = dd.DataDesignerConfigBuilder()

# --- Sampler columns ---

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="category",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["Electronics", "Clothing", "Books", "Home"],
            weights=[0.3, 0.25, 0.25, 0.2],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="customer",
        sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
        params=dd.PersonFromFakerSamplerParams(locale="en_US"),
        drop=True,  # keep derived fields, drop raw person object
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="rating",
        sampler_type=dd.SamplerType.UNIFORM,
        params=dd.UniformSamplerParams(low=1, high=5),
        convert_to="int",
    )
)

# --- Expression column (derived from sampler) ---

config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="customer_name",
        expr="{{ customer.first_name }} {{ customer.last_name }}",
    )
)

# --- LLM text columns ---

config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="product_name",
        prompt=("Create a creative product name for a {{ category }} product. Respond with only the product name."),
        model_alias="nvidia-text",
    )
)

config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="review",
        prompt=(
            "You are {{ customer_name }}, a {{ customer.age }}-year-old from "
            "{{ customer.city }}. Write a {{ rating }}-star review for "
            "{{ product_name }}. Be authentic and detailed."
        ),
        model_alias="nvidia-text",
    )
)

# --- Preview then create ---

preview = data_designer.preview(config_builder, num_records=3)
preview.display_sample_record()

results = data_designer.create(config_builder, num_records=100, dataset_name="product-reviews")
dataset = results.load_dataset()
analysis = results.load_analysis()
