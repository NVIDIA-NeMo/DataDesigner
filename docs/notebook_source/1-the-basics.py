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
# # 🎨 Data Designer Tutorial: The Basics
#
# #### 📚 What you'll learn
#
# This notebook demonstrates the basics of Data Designer by generating a simple product review dataset.
#

# %% [markdown]
# ### 📦 Import the essentials
#
# - The `essentials` module provides quick access to the most commonly used objects.
#

# %%
from data_designer.essentials import (
    CategorySamplerParams,
    ChatCompletionInferenceParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    ModelConfig,
    PersonFromFakerSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
)

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# - `DataDesigner` is the main object is responsible for managing the data generation process.
#
# - When initialized without arguments, the [default model providers](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) are used.
#

# %%
data_designer = DataDesigner()

# %% [markdown]
# ### 🎛️ Define model configurations
#
# - Each `ModelConfig` defines a model that can be used during the generation process.
#
# - The "model alias" is used to reference the model in the Data Designer config (as we will see below).
#
# - The "model provider" is the external service that hosts the model (see the [model config](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) docs for more details).
#
# - By default, we use [build.nvidia.com](https://build.nvidia.com/models) as the model provider.
#

# %%
# This name is set in the model provider configuration.
MODEL_PROVIDER = "nvidia"

# The model ID is from build.nvidia.com.
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"

# We choose this alias to be descriptive for our use case.
MODEL_ALIAS = "nemotron-nano-v2"

# This sets reasoning to False for the nemotron-nano-v2 model.
SYSTEM_PROMPT = "/no_think"

model_configs = [
    ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=ChatCompletionInferenceParams(
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
# ## 🎲 Getting started with sampler columns
#
# - Sampler columns offer non-LLM based generation of synthetic data.
#
# - They are particularly useful for **steering the diversity** of the generated data, as we demonstrate below.
#
# <br>
#
# You can view available samplers using the config builder's `info` property:
#

# %%
config_builder.info.display("samplers")

# %% [markdown]
# Let's start designing our product review dataset by adding product category and subcategory columns.
#

# %%
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

# Optionally validate that the columns are configured correctly.
config_builder.validate()

# %% [markdown]
# Next, let's add samplers to generate data related to the customer and their review.
#

# %%
config_builder.add_column(
    SamplerColumnConfig(
        name="customer",
        sampler_type=SamplerType.PERSON_FROM_FAKER,
        params=PersonFromFakerSamplerParams(age_range=[18, 70], locale="en_US"),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="number_of_stars",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=1, high=5),
        convert_to="int",  # Convert the sampled float to an integer.
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="review_style",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["rambling", "brief", "detailed", "structured with bullet points"],
            weights=[1, 2, 2, 1],
        ),
    )
)

config_builder.validate()

# %% [markdown]
# ## 🦜 LLM-generated columns
#
# - The real power of Data Designer comes from leveraging LLMs to generate text, code, and structured data.
#
# - When prompting the LLM, we can use Jinja templating to reference other columns in the dataset.
#
# - As we see below, nested json fields can be accessed using dot notation.
#

# %%
config_builder.add_column(
    LLMTextColumnConfig(
        name="product_name",
        prompt=(
            "You are a helpful assistant that generates product names. DO NOT add quotes around the product name.\n\n"
            "Come up with a creative product name for a product in the '{{ product_category }}' category, focusing "
            "on products related to '{{ product_subcategory }}'. The target age range of the ideal customer is "
            "{{ target_age_range }} years old. Respond with only the product name, no other text."
        ),
        system_prompt=SYSTEM_PROMPT,
        model_alias=MODEL_ALIAS,
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="customer_review",
        prompt=(
            "You are a customer named {{ customer.first_name }} from {{ customer.city }}, {{ customer.state }}. "
            "You are {{ customer.age }} years old and recently purchased a product called {{ product_name }}. "
            "Write a review of this product, which you gave a rating of {{ number_of_stars }} stars. "
            "The style of the review should be '{{ review_style }}'."
        ),
        system_prompt=SYSTEM_PROMPT,
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
preview = data_designer.preview(config_builder, num_records=2)

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
results = data_designer.create(config_builder, num_records=10)

# %%
# Load the generated dataset as a pandas DataFrame.
dataset = results.load_dataset()

dataset.head()

# %%
# Load the analysis results into memory.
analysis = results.load_analysis()

analysis.to_report()

# %% [markdown]
# ## ⏭️ Next Steps
#
# Now that you've seen the basics of Data Designer, check out the following notebooks to learn more about:
#
# - [Structured outputs and jinja expressions](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/2-structured-outputs-and-jinja-expressions/)
#
# - [Seeding synthetic data generation with an external dataset](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/3-seeding-with-a-dataset/)
#
# - [Providing images as context](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/4-providing-images-as-context/)
#
