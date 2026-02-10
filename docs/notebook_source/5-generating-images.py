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
# # 🎨 Data Designer Tutorial: Generating Images
#
# #### 📚 What you'll learn
#
# This notebook shows how to generate synthetic image data with Data Designer using image-generation models.
#
# - 🖼️ **Image generation columns**: Add columns that produce images from text prompts
# - 📝 **Jinja2 prompts**: Drive diversity by referencing other columns in your prompt template
# - 💾 **Preview vs create**: Preview stores base64 in the dataframe; create saves images to disk and stores paths
#
# Data Designer supports both **diffusion** (e.g. DALL·E, Stable Diffusion, Imagen) and **autoregressive** (e.g. Gemini image, GPT image) models; the API is chosen automatically from the model name.
#
# If this is your first time using Data Designer, we recommend starting with the [first notebook](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/1-the-basics/) in this tutorial series.
#

# %% [markdown]
# ### 📦 Import Data Designer
#
# - `data_designer.config` provides the configuration API.
# - `DataDesigner` is the main interface for generation.
#

# %%
from IPython.display import Image as IPImage
from IPython.display import display

import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# When initialized without arguments, [default model providers](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) are used. This tutorial uses [OpenRouter](https://openrouter.ai) with the Flux 2 Pro image model; set `OPENROUTER_API_KEY` in your environment.
#

# %%
data_designer = DataDesigner()

# %% [markdown]
# ### 🎛️ Define an image-generation model
#
# - Use `ImageInferenceParams` so Data Designer treats this model as an image generator.
# - Image options (size, quality, aspect ratio, etc.) are model-specific; pass them via `extra_body`.
#

# %%
MODEL_PROVIDER = "openrouter"
MODEL_ID = "black-forest-labs/flux.2-pro"
MODEL_ALIAS = "image-model"

model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ImageInferenceParams(
            extra_body={"height": 512, "width": 512},
        ),
    )
]

# %% [markdown]
# ### 🏗️ Build the config: samplers + image column
#
# We'll generate diverse **dog portrait** images: sampler columns drive subject (breed), age, style, look direction, and emotion. The image-generation column uses a Jinja2 prompt that references all of them.
#

# %%
config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="style",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "photorealistic",
                "oil painting",
                "watercolor",
                "digital art",
                "sketch",
                "anime",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="dog_breed",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "a Golden Retriever",
                "a German Shepherd",
                "a Labrador Retriever",
                "a Bulldog",
                "a Beagle",
                "a Poodle",
                "a Corgi",
                "a Siberian Husky",
                "a Dalmatian",
                "a Yorkshire Terrier",
                "a Boxer",
                "a Dachshund",
                "a Doberman Pinscher",
                "a Shih Tzu",
                "a Chihuahua",
                "a Border Collie",
                "an Australian Shepherd",
                "a Cocker Spaniel",
                "a Maltese",
                "a Pomeranian",
                "a Saint Bernard",
                "a Great Dane",
                "an Akita",
                "a Samoyed",
                "a Boston Terrier",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="cat_breed",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "a Persian",
                "a Maine Coon",
                "a Siamese",
                "a Ragdoll",
                "a Bengal",
                "an Abyssinian",
                "a British Shorthair",
                "a Sphynx",
                "a Scottish Fold",
                "a Russian Blue",
                "a Birman",
                "an Oriental Shorthair",
                "a Norwegian Forest Cat",
                "a Devon Rex",
                "a Burmese",
                "an Egyptian Mau",
                "a Tonkinese",
                "a Himalayan",
                "a Savannah",
                "a Chartreux",
                "a Somali",
                "a Manx",
                "a Turkish Angora",
                "a Balinese",
                "an American Shorthair",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="dog_age",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["1-3", "3-6", "6-9", "9-12", "12-15"],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="cat_age",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["1-3", "3-6", "6-9", "9-12", "12-18"],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="dog_look_direction",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["left", "right", "front", "up", "down"],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="cat_look_direction",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["left", "right", "front", "up", "down"],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="dog_emotion",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["happy", "curious", "serious", "sleepy", "excited"],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="cat_emotion",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["aloof", "curious", "content", "sleepy", "playful"],
        ),
    )
)

config_builder.add_column(
    dd.ImageColumnConfig(
        name="generated_image",
        prompt=(
            """
A {{ style }} family pet portrait of a {{ dog_breed }} dog of {{ dog_age }} years old looking {{dog_look_direction}} with an {{ dog_emotion }} expression and
{{ cat_breed }} cat of {{ cat_age }} years old looking {{ cat_look_direction }} with an {{ cat_emotion }} expression in the background. Both subjects should be in focus.
        """
        ),
        model_alias=MODEL_ALIAS,
    )
)

data_designer.validate(config_builder)

# %% [markdown]
# ### 🔁 Preview: images as base64
#
# In **preview** mode, generated images are stored as base64 strings in the dataframe. Run the next cell to step through each record (images are shown in the sample record display, but only in a notebook environment).
#

# %%
preview = data_designer.preview(config_builder, num_records=2)

# %%
for i in range(len(preview.dataset)):
    preview.display_sample_record()

# %%
preview.dataset

# %% [markdown]
# ### 🆙 Create: images saved to disk
#
# In **create** mode, images are written to an `images/` folder with UUID filenames; the dataframe stores relative paths (e.g. `images/1d16b6e2-562f-4f51-91e5-baaa999ea916.png`).
#

# %%
results = data_designer.create(config_builder, num_records=5, dataset_name="tutorial-5-images")

# %%
dataset = results.load_dataset()
dataset.head()

# %%
# Display all image from the created dataset. Paths are relative to the artifact output directory.
for index, row in dataset.iterrows():
    path_or_list = row.get("generated_image")
    if path_or_list is not None:
        for path in path_or_list:
            base = results.artifact_storage.base_dataset_path
            full_path = base / path
            display(IPImage(data=full_path))

# %% [markdown]
# ## ⏭️ Next steps
#
# - [The basics](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/1-the-basics/): samplers and LLM text columns
# - [Structured outputs and Jinja](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/2-structured-outputs-and-jinja-expressions/)
# - [Seeding with a dataset](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/3-seeding-with-a-dataset/)
# - [Providing images as context](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/4-providing-images-as-context/)
#
