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
# # 🎨 Data Designer 101: Providing Images as Context for Multi-Modal Synthetic Data Generation

# %% [markdown]
# #### 📚 What you'll learn
#
# This notebook demonstrates how to provide images as context for a more complex multi-modal synthetic Question-Answer dataset generation workflow from visual documents.
#
# - ✨ **Visual Document Processing**: Converting images to chat-ready format
# - 🏗️ **Structured Output Generation**: Using Pydantic models for consistent data schemas
# - 🎯 **Multi-step Generation Pipeline**: Summary → Question → Answer generation workflow

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
from typing import Literal
import uuid

from datasets import load_dataset
from IPython.display import display

# Third-party imports
import pandas as pd
from pydantic import BaseModel, Field
import rich
from rich.panel import Panel

# Data Designer imports
from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    ImageContext,
    ImageFormat,
    InferenceParameters,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    ModalityDataType,
    ModelConfig,
    SamplerColumnConfig,
    SamplerType,
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
        alias="text",
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(temperature=0.85, top_p=0.95, max_tokens=1024),
    ),
    ModelConfig(
        alias="vision",
        model="nvidia/nemotron-nano-12b-v2-vl",
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
# In this section, we'll prepare our visual documents as a seed dataset. The seed dataset provides the foundation for synthetic data generation by:
#
# - **Loading Visual Documents**: We use the ColPali dataset containing document images
# - **Image Processing**: Convert images to base64 format for model consumption
# - **Metadata Extraction**: Preserve relevant document information
# - **Sampling Strategy**: Configure how the seed data is utilized during generation
#
# The seed dataset can be referenced in generation prompts using Jinja templating.

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
# ### 🎨 Designing our Data Schema
#
# Structured outputs ensure consistent and predictable data generation. Data Designer supports schemas defined using:
# - **JSON Schema**: For basic structure definition
# - **Pydantic Models**: For advanced validation and type safety (recommended)
#
# We'll use Pydantic models to define our Question-Answer schema:
#


# %%
class Question(BaseModel):
    """Schema for generated questions"""

    question: str = Field(description="The question to be generated")


class QuestionTopic(BaseModel):
    """Schema for question topics"""

    topic: str = Field(description="The topic/category of the question")


class Options(BaseModel):
    """Schema for multiple choice options"""

    option_a: str = Field(description="The first answer choice")
    option_b: str = Field(description="The second answer choice")
    option_c: str = Field(description="The third answer choice")
    option_d: str = Field(description="The fourth answer choice")


class Answer(BaseModel):
    """Schema for question answers"""

    answer: Literal["option_a", "option_b", "option_c", "option_d"] = Field(
        description="The correct answer to the question"
    )


# %%
config_builder.add_column(
    SamplerColumnConfig(
        name="difficulty",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=["easy", "medium", "hard"]),
    )
)

config_builder.add_column(
    LLMStructuredColumnConfig(
        name="question",
        model_alias="text",
        prompt=(
            "Generate a question based on the following context: {{ summary }}. "
            "The difficulty of the generated question should be {{ difficulty }}"
        ),
        system_prompt=(
            "You are a helpful assistant that generates questions based on the given context. "
            "The context are sourced from documents pertaining to the petroleum industry. "
            "You will be given a context and you will need to generate a question based on the context. "
            "The difficulty of the generated question should be {{ difficulty }}. "
            "Ensure you generate just the question and no other text."
        ),
        output_format=Question,
    )
)

config_builder.add_column(
    LLMStructuredColumnConfig(
        name="options",
        model_alias="text",
        prompt=(
            "Generate four answer choices for the question: {{ question }} based on the following context: {{ summary }}. "
            "The option you generate should match the difficulty of the generated question, {{ difficulty }}."
        ),
        output_format=Options,
    )
)


config_builder.add_column(
    LLMStructuredColumnConfig(
        name="answer",
        model_alias="text",
        prompt=(
            "Choose the correct answer for the question: {{ question }} based on the following context: {{ summary }} "
            "and options choices. The options are {{ options }}. Only select one of the options as the answer."
        ),
        output_format=Answer,
    )
)


config_builder.add_column(
    LLMStructuredColumnConfig(
        name="topic",
        model_alias="text",
        prompt=(
            "Generate the topic of the question: {{ question }} based on the following context: {{ summary }}. "
            "The topic should be a single word or phrase that is relevant to the question and context."
        ),
        system_prompt=(
            "Generate a short 1-3 word topic for the question: {{ question }} based on the given context. {{ summary }}"
        ),
        output_format=QuestionTopic,
    )
)


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
preview = data_designer.preview(config_builder, num_records=10)

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
# ### 🔎 Visual Inspection
#
# Let's compare the original document image with the generated outputs to validate quality:
#

# %%
# Compare original document with generated outputs
index = 0  # Change this to view different examples

# Merge preview data with original images for comparison
comparison_dataset = preview.dataset.merge(pd.DataFrame(img_dataset)[["uuid", "image"]], how="left", on="uuid")

# Extract the record for display
record = comparison_dataset.iloc[index]

print("📄 Original Document Image:")
display(resize_image(record.image, BASE64_IMAGE_HEIGHT))

print("\n📝 Generated Summary:")
rich.print(Panel(record.summary, title="Document Summary", title_align="left"))

print("\n❓ Generated Question:")
question_text = record.question.get("question") if isinstance(record.question, dict) else record.question
rich.print(Panel(str(question_text), title=f"Question (Difficulty: {record.difficulty})", title_align="left"))

print("\n🔢 Generated Options:")
options = record.options
if isinstance(options, dict):
    options_text = "\n".join(
        [
            f"A) {options.get('option_a', 'N/A')}",
            f"B) {options.get('option_b', 'N/A')}",
            f"C) {options.get('option_c', 'N/A')}",
            f"D) {options.get('option_d', 'N/A')}",
        ]
    )
else:
    options_text = str(options)
rich.print(Panel(options_text, title="Answer Choices", title_align="left"))

print("\n✅ Generated Answer:")
answer = record.answer.get("answer") if isinstance(record.answer, dict) else record.answer
rich.print(Panel(str(answer).upper().replace("_", " "), title="Correct Answer", title_align="left"))

print("\n🏷️ Topic:")
topic = record.topic.get("topic") if isinstance(record.topic, dict) else record.topic
rich.print(Panel(str(topic), title="Question Topic", title_align="left"))


# %% [markdown]
# ### 🆙 Scale up!
#
# - Happy with your preview data?
#
# - Use the `create` method to submit larger Data Designer generation jobs.
#

# %%
results = data_designer.create(config_builder, num_records=20)

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
# Now that you've seen how to use visual context in Data Designer, explore more:
#
# - Experiment with different vision models for specific document types
# - Apply this pattern to other vision-based tasks like image captioning or understanding
#
