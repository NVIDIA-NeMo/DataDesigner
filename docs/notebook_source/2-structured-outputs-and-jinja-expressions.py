# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎨 Data Designer Tutorial: Structured Outputs and Jinja Expressions
#
# #### 📚 What you'll learn
#
# In this notebook, we will continue our exploration of Data Designer, demonstrating more advanced data generation using structured outputs and Jinja expressions.
#
# If this is your first time using Data Designer, we recommend starting with the [first notebook](/notebooks/1-the-basics/) in this tutorial series.
#

# %% [markdown]
# ### 📦 Import the essentials
#
# - The `essentials` module provides quick access to the most commonly used objects.
#

# %%
from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    ExpressionColumnConfig,
    InferenceParameters,
    LLMStructuredColumnConfig,
    ModelConfig,
    PersonFromFakerSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
)

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# - `DataDesigner` is the main object that is used to interface with the library.
#
# - When initialized without arguments, the [default model providers](https://nvidia-nemo.github.io/DataDesigner/concepts/models/default-model-settings/) are used.
#

# %%
data_designer_client = DataDesigner()

# %% [markdown]
# ### 🎛️ Define model configurations
#
# - Each `ModelConfig` defines a model that can be used during the generation process.
#
# - The "model alias" is used to reference the model in the Data Designer config (as we will see below).
#
# - The "model provider" is the external service that hosts the model (see the [model config](https://nvidia-nemo.github.io/DataDesigner/concepts/models/default-model-settings/) docs for more details).
#
# - By default, we use [build.nvidia.com](https://build.nvidia.com/models) as the model provider.
#

# %%
# This name is set in the model provider configuration.
MODEL_PROVIDER = "nvidia"

# The model ID is from build.nvidia.com.
MODEL_ID = "nvidia/nvidia-nemotron-nano-9b-v2"

# We choose this alias to be descriptive for our use case.
MODEL_ALIAS = "nemotron-nano-v2"

# This sets reasoning to False for the nemotron-nano-v2 model.
SYSTEM_PROMPT = "/no_think"

model_configs = [
    ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(
            temperature=0.5,
            top_p=1.0,
            max_tokens=1024,
        ),
    )
]

# %% [markdown]
# ### 🏗️ Initialize the Data Designer Config Builder
#
# - The Data Designer config defines the dataset schema and generation process.
#
# - The config builder provides an intuitive interface for building this configuration.
#
# - The list of model configs is provided to the builder at initialization.
#

# %%
config_builder = DataDesignerConfigBuilder(model_configs=model_configs)

# %% [markdown]
# ### 🧑‍🎨 Designing our data
#
# - We will again create a product review dataset, but this time we will use structured outputs and Jinja expressions.
#
# - Structured outputs let you specify the exact schema of the data you want to generate.
#
# - Data Designer supports schemas specified using either json schema or Pydantic data models (recommended).
#
# <br>
#
# We'll define our structured outputs using [Pydantic](https://docs.pydantic.dev/latest/) data models
#
# > 💡 **Why Pydantic?**
# >
# > - Pydantic models provide better IDE support and type validation.
# >
# > - They are more Pythonic than raw JSON schemas.
# >
# > - They integrate seamlessly with Data Designer's structured output system.
#

# %%
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


# We define a Product schema so that the name, description, and price are generated
# in one go, with the types and constraints specified.
class Product(BaseModel):
    name: str = Field(description="The name of the product")
    description: str = Field(description="A description of the product")
    price: Decimal = Field(description="The price of the product", ge=10, le=1000, decimal_places=2)


class ProductReview(BaseModel):
    rating: int = Field(description="The rating of the product", ge=1, le=5)
    customer_mood: Literal["irritated", "mad", "happy", "neutral", "excited"] = Field(
        description="The mood of the customer"
    )
    review: str = Field(description="A review of the product")


# %% [markdown]
# Next, let's design our product review dataset using a few more tricks compared to the previous notebook.
#

# %%
# Since we often only want a few attributes from Person objects, we can
# set drop=True in the column config to drop the column from the final dataset.
config_builder.add_column(
    SamplerColumnConfig(
        name="customer",
        sampler_type=SamplerType.PERSON_FROM_FAKER,
        params=PersonFromFakerSamplerParams(),
        drop=True,
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="product_category",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Electronics",
                "Clothing",
                "Home & Kitchen",
                "Books",
                "Home Office",
            ],
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="product_subcategory",
        sampler_type=SamplerType.SUBCATEGORY,
        params=SubcategorySamplerParams(
            category="product_category",
            values={
                "Electronics": [
                    "Smartphones",
                    "Laptops",
                    "Headphones",
                    "Cameras",
                    "Accessories",
                ],
                "Clothing": [
                    "Men's Clothing",
                    "Women's Clothing",
                    "Winter Coats",
                    "Activewear",
                    "Accessories",
                ],
                "Home & Kitchen": [
                    "Appliances",
                    "Cookware",
                    "Furniture",
                    "Decor",
                    "Organization",
                ],
                "Books": [
                    "Fiction",
                    "Non-Fiction",
                    "Self-Help",
                    "Textbooks",
                    "Classics",
                ],
                "Home Office": [
                    "Desks",
                    "Chairs",
                    "Storage",
                    "Office Supplies",
                    "Lighting",
                ],
            },
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="target_age_range",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=["18-25", "25-35", "35-50", "50-65", "65+"]),
    )
)

# Sampler columns support conditional params, which are used if the condition is met.
# In this example, we set the review style to rambling if the target age range is 18-25.
# Note conditional parameters are only supported for Sampler column types.
config_builder.add_column(
    SamplerColumnConfig(
        name="review_style",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["rambling", "brief", "detailed", "structured with bullet points"],
            weights=[1, 2, 2, 1],
        ),
        conditional_params={
            "target_age_range == '18-25'": CategorySamplerParams(values=["rambling"]),
        },
    )
)

# Optionally validate that the columns are configured correctly.
config_builder.validate()

# %% [markdown]
# Next, we will use more advanced Jinja expressions to create new columns.
#
# Jinja expressions let you:
#
# - Access nested attributes: `{{ customer.first_name }}`
#
# - Combine values: `{{ customer.first_name }} {{ customer.last_name }}`
#
# - Use conditional logic: `{% if condition %}...{% endif %}`
#

# %%
# We can create new columns using Jinja expressions that reference
# existing columns, including attributes of nested objects.
config_builder.add_column(
    ExpressionColumnConfig(name="customer_name", expr="{{ customer.first_name }} {{ customer.last_name }}")
)

config_builder.add_column(ExpressionColumnConfig(name="customer_age", expr="{{ customer.age }}"))

config_builder.add_column(
    LLMStructuredColumnConfig(
        name="product",
        prompt=(
            "Create a product in the '{{ product_category }}' category, focusing on products  "
            "related to '{{ product_subcategory }}'. The target age range of the ideal customer is "
            "{{ target_age_range }} years old. The product should be priced between $10 and $1000."
        ),
        system_prompt=SYSTEM_PROMPT,
        output_format=Product,
        model_alias=MODEL_ALIAS,
    )
)

# We can even use if/else logic in our Jinja expressions to create more complex prompt patterns.
config_builder.add_column(
    LLMStructuredColumnConfig(
        name="customer_review",
        prompt=(
            "Your task is to write a review for the following product:\n\n"
            "Product Name: {{ product.name }}\n"
            "Product Description: {{ product.description }}\n"
            "Price: {{ product.price }}\n\n"
            "Imagine your name is {{ customer_name }} and you are from {{ customer.city }}, {{ customer.state }}. "
            "Write the review in a style that is '{{ review_style }}'."
            "{% if target_age_range == '18-25' %}"
            "Make sure the review is more informal and conversational."
            "{% else %}"
            "Make sure the review is more formal and structured."
            "{% endif %}"
        ),
        system_prompt=SYSTEM_PROMPT,
        output_format=ProductReview,
        model_alias=MODEL_ALIAS,
    )
)

config_builder.validate()

# %% [markdown]
# ### 🔁 Iteration is key – preview the dataset!
#
# 1. Use the `preview` method to generate a sample of records quickly.
#
# 2. Inspect the results for quality and format issues.
#
# 3. Adjust column configurations, prompts, or parameters as needed.
#
# 4. Re-run the preview until satisfied.
#

# %%
preview = data_designer_client.preview(config_builder, num_records=2)

# %%
# Run this cell multiple times to cycle through the 2 preview records.
preview.display_sample_record()

# %%
# The preview dataset is available as a pandas DataFrame.
preview.dataset

# %% [markdown]
# ### 📊 Analyze the generated data
#
# - Data Designer automatically generates a basic statistical analysis of the generated data.
#
# - This analysis is available via the `analysis` property of generation result objects.
#

# %%
# Print the analysis as a table.
preview.analysis.to_report()

# %% [markdown]
# ### 🆙 Scale up!
#
# - Happy with your preview data?
#
# - Use the `create` method to submit larger Data Designer generation jobs.
#

# %%
job_results = data_designer_client.create(config_builder, num_records=10)

# %%
# Load the generated dataset as a pandas DataFrame.
dataset = job_results.load_dataset()

dataset.head()

# %%
# Load the analysis results into memory.
analysis = job_results.load_analysis()

analysis.to_report()

# %% [markdown]
# ## ⏭️ Next Steps
#
# Check out the following notebook to learn more about:
#
# - [Seeding synthetic data generation with an external dataset](/notebooks/3-seeding-with-a-dataset/)
#
