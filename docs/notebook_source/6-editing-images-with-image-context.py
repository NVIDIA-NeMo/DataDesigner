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
# This notebook shows how to edit existing images by combining a seed dataset with image generation. You'll load animal portrait photographs from HuggingFace, feed them as context to an autoregressive model, and generate fun edited versions with accessories like sunglasses, top hats, and bow ties.
#
# - 🌱 **Seed datasets with images**: Load a HuggingFace image dataset and use it as a seed
# - 🖼️ **Image context for editing**: Pass existing images to an image-generation model via `multi_modal_context`
# - 🎲 **Sampler-driven diversity**: Combine sampled accessories and settings with seed images for varied results
# - 💾 **Preview vs create**: Preview stores base64 in the dataframe; create saves images to disk
#
# This tutorial uses an **autoregressive** model (one that supports both image input *and* image output via the chat completions API). Diffusion models (DALL·E, Stable Diffusion, etc.) do not support image context—see [Tutorial 5](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/5-generating-images/) for text-to-image generation with diffusion models.
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
import io
import uuid

import pandas as pd
from datasets import load_dataset
from IPython.display import Image as IPImage
from IPython.display import display

import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# We initialize Data Designer without arguments here—the image-editing model is configured explicitly in the next cell. No default text model is needed for this tutorial.
#

# %%
data_designer = DataDesigner()

# %% [markdown]
# ### 🎛️ Define an image-editing model
#
# We need an **autoregressive** model that supports both image input and image output via the chat completions API. This lets us pass existing images as context and receive edited images back.
#
# - Use `ImageInferenceParams` so Data Designer treats this model as an image generator.
# - Image-specific options are model-dependent; pass them via `extra_body`.
#
# > **Note**: This tutorial uses the Flux 2 Pro model via [OpenRouter](https://openrouter.ai). Set `OPENROUTER_API_KEY` in your environment.
#

# %%
MODEL_PROVIDER = "openrouter"
MODEL_ID = "black-forest-labs/flux.2-pro"
MODEL_ALIAS = "image-editor"

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
# ### 🌱 Load animal portraits from HuggingFace
#
# We'll load animal face photographs from the [AFHQ](https://huggingface.co/datasets/huggan/AFHQv2) (Animal Faces-HQ) dataset, convert them to base64, and use them as a seed dataset.
#
# AFHQ contains high-quality 512×512 close-up portraits of cats, dogs, and wildlife—perfect subjects for adding fun accessories.
#

# %%
SEED_COUNT = 10
BASE64_IMAGE_HEIGHT = 512

ANIMAL_LABELS = {0: "cat", 1: "dog", 2: "wild"}


def resize_image(image, height: int):
    """Resize image maintaining aspect ratio."""
    original_width, original_height = image.size
    width = int(original_width * (height / original_height))
    return image.resize((width, height))


def prepare_record(record: dict, height: int) -> dict:
    """Convert a HuggingFace record to base64 with metadata."""
    image = resize_image(record["image"], height)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    base64_string = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return {
        "uuid": str(uuid.uuid4()),
        "base64_image": base64_string,
        "animal": ANIMAL_LABELS[record["label"]],
    }


# %%
print("📥 Streaming animal portraits from HuggingFace...")
hf_dataset = load_dataset("huggan/AFHQv2", split="train", streaming=True)

hf_iter = iter(hf_dataset)
records = [prepare_record(next(hf_iter), BASE64_IMAGE_HEIGHT) for _ in range(SEED_COUNT)]
df_seed = pd.DataFrame(records)

print(f"✅ Prepared {len(df_seed)} animal portraits with columns: {list(df_seed.columns)}")
df_seed.head()

# %% [markdown]
# ### 🏗️ Build the configuration
#
# We combine three ingredients:
#
# 1. **Seed dataset** — original animal portraits as base64 and their species labels
# 2. **Sampler columns** — randomly sample accessories and settings for each image
# 3. **Image column with context** — generate an edited image using the original as reference
#
# The `multi_modal_context` parameter on `ImageColumnConfig` tells Data Designer to pass the seed image to the model alongside the text prompt. The model receives both the image and the editing instructions, and generates a new image.
#

# %%
config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# 1. Seed the original animal portraits
config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=df_seed))

# 2. Add sampler columns for accessory diversity
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

# 3. Image column that reads the seed image as context and generates an edited version
config_builder.add_column(
    dd.ImageColumnConfig(
        name="edited_image",
        prompt=(
            "Edit this {{ animal }} portrait photo. "
            "Add {{ accessory }} on the animal. "
            "Place the {{ animal }} in {{ setting }}. "
            "Render the result in {{ art_style }}. "
            "Keep the animal's face, expression, and features faithful to the original photo."
        ),
        model_alias=MODEL_ALIAS,
        multi_modal_context=[
            dd.ImageContext(
                column_name="base64_image",
                data_type=dd.ModalityDataType.BASE64,
                image_format=dd.ImageFormat.PNG,
            )
        ],
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
# Let's display the original animal portraits next to their accessorized versions.
#


# %%
def display_before_after(row: pd.Series, index: int, base_path=None) -> None:
    """Display original vs edited image for a single record.

    When base_path is None (preview mode), edited_image is decoded from base64.
    When base_path is provided (create mode), edited_image is loaded from disk.
    """
    print(f"\n{'=' * 60}")
    print(f"Record {index}: {row['animal']} wearing {row['accessory']}")
    print(f"Setting: {row['setting']}")
    print(f"Style: {row['art_style']}")
    print(f"{'=' * 60}")

    print("\n📷 Original portrait:")
    display(IPImage(data=base64.b64decode(row["base64_image"])))

    print("\n🎨 Edited version:")
    edited = row.get("edited_image")
    if edited is None:
        return
    if base_path is None:
        images = edited if isinstance(edited, list) else [edited]
        for img_b64 in images:
            display(IPImage(data=base64.b64decode(img_b64)))
    else:
        paths = edited if not isinstance(edited, str) else [edited]
        for path in paths:
            display(IPImage(filename=str(base_path / path)))


# %%
for index, row in preview.dataset.iterrows():
    display_before_after(row, index)

# %% [markdown]
# ### 🆙 Create at scale
#
# In **create** mode, images are saved to disk in an `images/<column_name>/` folder with UUID filenames. The dataframe stores relative paths.
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
# - Experiment with different autoregressive models for image editing
# - Try more creative editing prompts (style transfer, background replacement, artistic filters)
# - Combine image editing with text generation (e.g., generate captions for edited images using an LLM-Text column)
#
# Related tutorials:
#
# - [The basics](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/1-the-basics/): samplers and LLM text columns
# - [Providing images as context](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/4-providing-images-as-context/): image-to-text with VLMs
# - [Generating images](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/5-generating-images/): text-to-image generation with diffusion models
#
