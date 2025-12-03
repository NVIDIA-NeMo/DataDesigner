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
# # 🎨 Data Designer 101: Seeding Synthetic Data Generation with an External Dataset
#
# #### 📚 What you'll learn
#
# In this notebook, we will demonstrate how to seed synthetic data generation in Data Designer with an external dataset.
#
# If this is your first time using Data Designer, we recommend starting with the [first notebook](/notebooks/1-the-basics/) in this 101 series.
#

# %% [markdown]
# ### 📦 Import the essentials
#
# - The `essentials` module provides quick access to the most commonly used objects.
#

# %%
from data_designer.essentials import (
    DataDesigner,
    DataDesignerConfigBuilder,
    InferenceParameters,
    ModelConfig,
    SeedConfig,
)

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# - `DataDesigner` is the main object is responsible for managing the data generation process.
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
# ## 🏥 Prepare a seed dataset
#
# - For this notebook, we'll create a synthetic dataset of patient notes.
#
# - We will _seed_ the generation process with a [symptom-to-diagnosis dataset](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis).
#
# - We already have the dataset downloaded in the [data](../data) directory of this repository.
#
# <br>
#
# > 🌱 **Why use a seed dataset?**
# >
# > - Seed datasets let you steer the generation process by providing context that is specific to your use case.
# >
# > - Seed datasets are also an excellent way to inject real-world diversity into your synthetic data.
# >
# > - During generation, prompt templates can reference any of the seed dataset fields.
#

# %%
# Download sample dataset from Github
import urllib.request

url = "https://raw.githubusercontent.com/NVIDIA/GenerativeAIExamples/refs/heads/main/nemo/NeMo-Data-Designer/data/gretelai_symptom_to_diagnosis.csv"
local_filename, headers = urllib.request.urlretrieve(url, "gretelai_symptom_to_diagnosis.csv")

seed_dataset = SeedConfig(dataset=local_filename)

# Pass the reference to the config builder for use during generation.
config_builder.with_seed_dataset(seed_dataset)

# %% [markdown]
# ## 🎨 Designing our synthetic patient notes dataset
#
# - Here we use `add_column` with keyword arguments (rather than imported config objects).
#
# - Generally, we recommend using concrete objects, but this is a convenient shorthand.
#
# - **Note**: The prompt template can reference fields from our seed dataset:
#   - `{{ diagnosis }}` - the medical diagnosis from the seed data
#   - `{{ patient_summary }}` - the symptom description from the seed data
#

# %%
config_builder.add_column(
    name="patient_sampler",
    column_type="sampler",
    sampler_type="person_from_faker",
)

config_builder.add_column(
    name="doctor_sampler",
    column_type="sampler",
    sampler_type="person_from_faker",
)

config_builder.add_column(
    name="patient_id",
    column_type="sampler",
    sampler_type="uuid",
    params={
        "prefix": "PT-",
        "short_form": True,
        "uppercase": True,
    },
)

config_builder.add_column(
    name="first_name",
    column_type="expression",
    expr="{{ patient_sampler.first_name}}",
)

config_builder.add_column(
    name="last_name",
    column_type="expression",
    expr="{{ patient_sampler.last_name }}",
)


config_builder.add_column(
    name="dob",
    column_type="expression",
    expr="{{ patient_sampler.birth_date }}",
)

config_builder.add_column(
    name="symptom_onset_date",
    column_type="sampler",
    sampler_type="datetime",
    params={"start": "2024-01-01", "end": "2024-12-31"},
)

config_builder.add_column(
    name="date_of_visit",
    column_type="sampler",
    sampler_type="timedelta",
    params={"dt_min": 1, "dt_max": 30, "reference_column_name": "symptom_onset_date"},
)

config_builder.add_column(
    name="physician",
    column_type="expression",
    expr="Dr. {{ doctor_sampler.last_name }}",
)

config_builder.add_column(
    name="physician_notes",
    column_type="llm-text",
    prompt="""\
You are a primary-care physician who just had an appointment with {{ first_name }} {{ last_name }},
who has been struggling with symptoms from {{ diagnosis }} since {{ symptom_onset_date }}.
The date of today's visit is {{ date_of_visit }}.

{{ patient_summary }}

Write careful notes about your visit with {{ first_name }},
as Dr. {{ doctor_sampler.first_name }} {{ doctor_sampler.last_name }}.

Format the notes as a busy doctor might.
""",
    model_alias=MODEL_ALIAS,
    system_prompt=SYSTEM_PROMPT,
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
preview = data_designer_client.preview(config_builder)

# %%
# Run this cell multiple times to cycle through the 10 preview records.
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
job_results = data_designer_client.create(config_builder, num_records=20)

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
# Use Data Designer to generate synthetic data for your specific use case!
#
