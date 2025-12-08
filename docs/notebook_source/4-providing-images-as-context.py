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
# # 🎨 Data Designer Tutorial: Providing Images as Context for Vision-Based Data Generation

# %% [markdown]
# #### 📚 What you'll learn
#
# This notebook demonstrates how to provide images as context to generate text descriptions using vision-language models.
#
# - ✨ **Visual Document Processing**: Converting images to chat-ready format for model consumption
# - 🔍 **Vision-Language Generation**: Using vision models to generate detailed summaries from images
#
# If this is your first time using Data Designer, we recommend starting with the [first notebook](/notebooks/1-the-basics/) in this tutorial series.
#

# %% [markdown]
# ### ⬇️ Install dependencies (if required)

# %%
# !uv pip install pillow

# %% [markdown]
# ### 📦 Import the essentials
#
# - The `essentials` module provides quick access to the most commonly used objects.
#

# %%
# Standard library imports
import base64
import io
import uuid

# Third-party imports
import pandas as pd
import rich
from datasets import load_dataset
from IPython.display import display
from rich.panel import Panel

# Data Designer imports
from data_designer.essentials import (
    DataDesigner,
    DataDesignerConfigBuilder,
    ImageContext,
    ImageFormat,
    InferenceParameters,
    LLMTextColumnConfig,
    ModalityDataType,
    ModelConfig,
)

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# - `DataDesigner` is the main object is responsible for managing the data generation process.
#
# - When initialized without arguments, the [default model providers](https://nvidia-nemo.github.io/DataDesigner/concepts/models/default-model-settings/) are used.
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
# - The "model provider" is the external service that hosts the model (see the [model config](https://nvidia-nemo.github.io/DataDesigner/concepts/models/default-model-settings/) docs for more details).
#
# - By default, we use [build.nvidia.com](https://build.nvidia.com/models) as the model provider.
#

# %%
# This name is set in the model provider configuration.
MODEL_PROVIDER = "nvidia"

model_configs = [
    ModelConfig(
        alias="vision",
        model="meta/llama-4-scout-17b-16e-instruct",
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(
            temperature=0.60,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
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
# ### 🌱 Seed Dataset Creation
#
# In this section, we'll prepare our visual documents as a seed dataset for summarization:
#
# - **Loading Visual Documents**: We use the ColPali dataset containing document images
# - **Image Processing**: Convert images to base64 format for vision model consumption
# - **Metadata Extraction**: Preserve relevant document information (filename, page number, source, etc.)
#
# The seed dataset will be used to generate detailed text summaries of each document image.

# %%
# Dataset processing configuration
IMG_COUNT = 512  # Number of images to process
BASE64_IMAGE_HEIGHT = 512  # Standardized height for model input

# Load ColPali dataset for visual documents
img_dataset_cfg = {"path": "vidore/colpali_train_set", "split": "train", "streaming": True}


# %%
def resize_image(image, height: int):
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image object
        height: Target height in pixels

    Returns:
        Resized PIL Image object
    """
    original_width, original_height = image.size
    width = int(original_width * (height / original_height))
    return image.resize((width, height))


def convert_image_to_chat_format(record, height: int) -> dict:
    """
    Convert PIL image to base64 format for chat template usage.

    Args:
        record: Dataset record containing image and metadata
        height: Target height for image resizing

    Returns:
        Updated record with base64_image and uuid fields
    """
    # Resize image for consistent processing
    image = resize_image(record["image"], height)

    # Convert to base64 string
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    byte_data = img_buffer.getvalue()
    base64_encoded_data = base64.b64encode(byte_data)
    base64_string = base64_encoded_data.decode("utf-8")

    # Return updated record
    return record | {"base64_image": base64_string, "uuid": str(uuid.uuid4())}


# %%
# Load and process the visual document dataset
print("📥 Loading and processing document images...")

img_dataset_iter = iter(
    load_dataset(**img_dataset_cfg).map(convert_image_to_chat_format, fn_kwargs={"height": BASE64_IMAGE_HEIGHT})
)
img_dataset = pd.DataFrame([next(img_dataset_iter) for _ in range(IMG_COUNT)])

print(f"✅ Loaded {len(img_dataset)} images with columns: {list(img_dataset.columns)}")

# %%
img_dataset.head()

# %%
# Add the seed dataset containing our processed images
df_seed = pd.DataFrame(img_dataset)[["uuid", "image_filename", "base64_image", "page", "options", "source"]]
config_builder.with_seed_dataset(
    DataDesigner.make_seed_reference_from_dataframe(df_seed, file_path="colpali_train_set.csv")
)

# %%
# Add a column to generate detailed document summaries
config_builder.add_column(
    LLMTextColumnConfig(
        name="summary",
        model_alias="vision",
        prompt=(
            "Provide a detailed summary of the content in this image in Markdown format. "
            "Start from the top of the image and then describe it from top to bottom. "
            "Place a summary at the bottom."
        ),
        multi_modal_context=[
            ImageContext(
                column_name="base64_image",
                data_type=ModalityDataType.BASE64,
                image_format=ImageFormat.PNG,
            )
        ],
    )
)


# %% [markdown]
#


# %% [markdown]
# ### 🔁 Iteration is key – preview the dataset!
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
# ### 🔎 Visual Inspection
#
# Let's compare the original document image with the generated summary to validate quality:
#

# %%
# Compare original document with generated summary
index = 0  # Change this to view different examples

# Merge preview data with original images for comparison
comparison_dataset = preview.dataset.merge(pd.DataFrame(img_dataset)[["uuid", "image"]], how="left", on="uuid")

# Extract the record for display
record = comparison_dataset.iloc[index]

print("📄 Original Document Image:")
display(resize_image(record.image, BASE64_IMAGE_HEIGHT))

print("\n📝 Generated Summary:")
rich.print(Panel(record.summary, title="Document Summary", title_align="left"))


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
# Now that you've learned how to use visual context for image summarization in Data Designer, explore more:
#
# - Experiment with different vision models for specific document types
# - Try different prompt variations to generate specialized descriptions (e.g., technical details, key findings)
# - Combine vision-based summaries with other column types for multi-modal workflows
# - Apply this pattern to other vision tasks like image captioning, OCR validation, or visual question answering
#
