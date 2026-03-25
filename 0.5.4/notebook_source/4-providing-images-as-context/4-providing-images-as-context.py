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
# If this is your first time using Data Designer, we recommend starting with the [first notebook](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/1-the-basics/) in this tutorial series.
#

# %% [markdown]
# ### 📦 Import Data Designer
#
# - `data_designer.config` provides access to the configuration API.
#
# - `DataDesigner` is the main interface for data generation.
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
import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% [markdown]
# ### ⚙️ Initialize the Data Designer interface
#
# - `DataDesigner` is the main object responsible for managing the data generation process.
#
# - When initialized without arguments, the [default model providers](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) are used.
#

# %%
data_designer = DataDesigner()

# %% [markdown]
# ### 🏗️ Initialize the Data Designer Config Builder
#
# - The Data Designer config defines the dataset schema and generation process.
#
# - The config builder provides an intuitive interface for building this configuration.
#
# - When initialized without arguments, the [default model configurations](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) are used.
#

# %%
config_builder = dd.DataDesignerConfigBuilder()

# %% [markdown]
# ### 🌱 Seed Dataset Creation
#
# In this section, we'll prepare our visual documents as a seed dataset for summarization:
#
# - **Loading Visual Documents**: We use a small pets image dataset containing labeled images
# - **Image Processing**: Convert images to base64 format for vision model consumption
# - **Metadata Extraction**: Preserve relevant image information (label, etc.)
#
# The seed dataset will be used to generate detailed text descriptions of each image.

# %%
# Dataset processing configuration
IMG_COUNT = 512  # Number of images to process
BASE64_IMAGE_HEIGHT = 512  # Standardized height for model input

# Load the pets dataset (train split, ~23 MB total)
img_dataset_cfg = {"path": "rokmr/pets", "split": "train"}


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
    image = resize_image(record["image"], height)

    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    byte_data = img_buffer.getvalue()
    base64_encoded_data = base64.b64encode(byte_data)
    base64_string = base64_encoded_data.decode("utf-8")

    return record | {"base64_image": base64_string, "uuid": str(uuid.uuid4())}


# %%
# Load and process the image dataset
print("📥 Loading and processing images...")

img_dataset = load_dataset(**img_dataset_cfg).map(
    convert_image_to_chat_format, fn_kwargs={"height": BASE64_IMAGE_HEIGHT}
)
img_dataset = pd.DataFrame(img_dataset[:IMG_COUNT])

print(f"✅ Loaded {len(img_dataset)} images with columns: {list(img_dataset.columns)}")

# %%
img_dataset.head()

# %%
# Add the seed dataset containing our processed images
df_seed = pd.DataFrame(img_dataset)[["uuid", "label", "base64_image"]]
config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=df_seed))

# %%
# Add a column to generate detailed image descriptions
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="description",
        model_alias="nvidia-vision",
        prompt=(
            "Provide a detailed description of the content in this image in Markdown format. "
            "Describe the main subject, background, colors, and any notable details."
        ),
        multi_modal_context=[dd.ImageContext(column_name="base64_image")],
    )
)

data_designer.validate(config_builder)

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
# Let's compare the original image with the generated description to validate quality:
#

# %%
# Compare original image with generated description
index = 0  # Change this to view different examples

# Merge preview data with original images for comparison
comparison_dataset = preview.dataset.merge(pd.DataFrame(img_dataset)[["uuid", "image"]], how="left", on="uuid")

# Extract the record for display
record = comparison_dataset.iloc[index]

print("📄 Original Image:")
display(resize_image(record.image, BASE64_IMAGE_HEIGHT))

print("\n📝 Generated Description:")
rich.print(Panel(record.description, title="Image Description", title_align="left"))


# %% [markdown]
# ### 🆙 Scale up!
#
# - Happy with your preview data?
#
# - Use the `create` method to submit larger Data Designer generation jobs.
#

# %%
results = data_designer.create(config_builder, num_records=10, dataset_name="tutorial-4")

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
# - Experiment with different vision models for specific image types
# - Try different prompt variations to generate specialized descriptions (e.g., technical details, key findings)
# - Combine vision-based descriptions with other column types for multi-modal workflows
# - Apply this pattern to other vision tasks like image captioning, OCR validation, or visual question answering
#
# - [Generating images](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/5-generating-images/) with Data Designer
#
