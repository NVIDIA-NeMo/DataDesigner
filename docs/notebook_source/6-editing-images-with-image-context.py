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
# # 🎨 Data Designer Tutorial: Image-to-Image Editing
#
# #### 📚 What you'll learn
#
# This notebook shows how to chain image generation columns: first generate animal portraits from text, then edit those generated images by adding accessories and changing styles—all without loading external datasets.
#
# - 🖼️ **Text-to-image generation**: Generate images from text prompts
# - 🔗 **Chaining image columns**: Use `ImageContext` to pass generated images to a follow-up editing column
# - 🎲 **Sampler-driven diversity**: Combine sampled accessories and settings for varied edits
#
# This tutorial uses an **autoregressive** model (one that supports both text-to-image *and* image-to-image generation via the chat completions API). Diffusion models (DALL·E, Stable Diffusion, etc.) do not support image context—see [Tutorial 5](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/5-generating-images/) for text-to-image generation with diffusion models.
#
# > **Prerequisites**: This tutorial uses [OpenRouter](https://openrouter.ai) with the Flux 2 Pro model. Set `OPENROUTER_API_KEY` in your environment before running.
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
import base64
from pathlib import Path

from IPython.display import Image as IPImage
from IPython.display import display

import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# We initialize Data Designer without arguments here—the image model is configured explicitly in the next cell.
#

# %%
data_designer = DataDesigner()

# %% [markdown]
# ### 🎛️ Define an image model
#
# We need an **autoregressive** model that supports both text-to-image and image-to-image generation via the chat completions API. This lets us generate images from text and then pass those images as context for editing.
#
# - Use `ImageInferenceParams` so Data Designer treats this model as an image generator.
# - Image-specific options are model-dependent; pass them via `extra_body`.
#
# > **Note**: This tutorial uses the Flux 2 Pro model via [OpenRouter](https://openrouter.ai). Set `OPENROUTER_API_KEY` in your environment.
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
# ### 🏗️ Build the configuration
#
# We chain two image generation columns:
#
# 1. **Sampler columns** — randomly sample animal types, accessories, settings, and art styles
# 2. **First image column** — generate an animal portrait from a text prompt
# 3. **Second image column with context** — edit the generated portrait using `ImageContext`
#

# %%
config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# 1. Sampler columns for diversity
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="animal",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["cat", "dog", "fox", "owl", "rabbit", "panda"],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="accessory",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "a tiny top hat",
                "oversized sunglasses",
                "a red bow tie",
                "a knitted beanie",
                "a flower crown",
                "a monocle and mustache",
                "a pirate hat and eye patch",
                "a chef hat",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="setting",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "a cozy living room",
                "a sunny park",
                "a photo studio with soft lighting",
                "a red carpet event",
                "a holiday card backdrop with snowflakes",
                "a tropical beach at sunset",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="art_style",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "a photorealistic style",
                "a Disney Pixar 3D render",
                "a watercolor painting",
                "a pop art poster",
            ],
        ),
    )
)

# 2. Generate animal portrait from text
config_builder.add_column(
    dd.ImageColumnConfig(
        name="animal_portrait",
        prompt="A close-up portrait photograph of a {{ animal }} looking at the camera, studio lighting, high quality.",
        model_alias=MODEL_ALIAS,
    )
)

# 3. Edit the generated portrait
config_builder.add_column(
    dd.ImageColumnConfig(
        name="edited_portrait",
        prompt=(
            "Edit this {{ animal }} portrait photo. "
            "Add {{ accessory }} on the animal. "
            "Place the {{ animal }} in {{ setting }}. "
            "Render the result in {{ art_style }}. "
            "Keep the animal's face, expression, and features faithful to the original photo."
        ),
        model_alias=MODEL_ALIAS,
        multi_modal_context=[dd.ImageContext(column_name="animal_portrait")],
    )
)

data_designer.validate(config_builder)

# %% [markdown]
# ### 🔁 Preview: quick iteration
#
# In **preview** mode, generated images are stored as base64 strings in the dataframe. Use this to iterate on your prompts, accessories, and sampler values before scaling up.
#

# %%
preview = data_designer.preview(config_builder, num_records=2)

# %%
for i in range(len(preview.dataset)):
    preview.display_sample_record()

# %%
preview.dataset

# %% [markdown]
# ### 🔎 Compare original vs edited
#
# Let's display the generated animal portraits next to their edited versions.
#


# %%
def display_image(image_value, base_path: Path | None = None) -> None:
    """Display an image from base64 (preview mode) or file path (create mode)."""
    values = image_value if isinstance(image_value, list) else [image_value]
    for value in values:
        if base_path is not None:
            display(IPImage(filename=str(base_path / value)))
        else:
            display(IPImage(data=base64.b64decode(value)))


def display_before_after(row, index: int, base_path: Path | None = None) -> None:
    """Display original portrait vs edited version for a single record."""
    print(f"\n{'=' * 60}")
    print(f"Record {index}: {row['animal']} wearing {row['accessory']}")
    print(f"Setting: {row['setting']}, Style: {row['art_style']}")
    print(f"{'=' * 60}")

    print("\n📷 Generated portrait:")
    display_image(row["animal_portrait"], base_path)

    print("\n🎨 Edited version:")
    display_image(row["edited_portrait"], base_path)


# %%
for index, row in preview.dataset.iterrows():
    display_before_after(row, index)

# %% [markdown]
# ### 🆙 Create at scale
#
# In **create** mode, images are saved to disk in `images/<column_name>/` folders with UUID filenames. The dataframe stores relative paths. `ImageContext` auto-detection handles this transparently—generated file paths are resolved to base64 before being sent to the model for editing.
#

# %%
results = data_designer.create(config_builder, num_records=5, dataset_name="tutorial-6-edited-images")

# %%
dataset = results.load_dataset()
dataset.head()

# %%
for index, row in dataset.head(10).iterrows():
    display_before_after(row, index, base_path=results.artifact_storage.base_dataset_path)

# %% [markdown]
# ## ⏭️ Next steps
#
# - Experiment with different autoregressive models for image generation and editing
# - Try more creative editing prompts (style transfer, background replacement, artistic filters)
# - Combine image generation with text generation (e.g., generate captions using an LLM-Text column with `ImageContext`)
# - Chain more than two image columns for multi-step editing pipelines
#
# Related tutorials:
#
# - [The basics](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/1-the-basics/): samplers and LLM text columns
# - [Providing images as context](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/4-providing-images-as-context/): image-to-text with VLMs
# - [Generating images](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/5-generating-images/): text-to-image generation with diffusion models
#
