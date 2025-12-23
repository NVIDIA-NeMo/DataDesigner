# Quick Start

Get started with Data Designer using the default model providers and configurations. Data Designer ships with built-in model providers and configurations that make it easy to start generating synthetic data immediately.

## Prerequisites

Before you begin, you'll need an API key from one of the default providers:

- **NVIDIA API Key**: Get yours from [build.nvidia.com](https://build.nvidia.com)
- **OpenAI API Key** (optional): Get yours from [platform.openai.com](https://platform.openai.com/api-keys)

Set your API key as an environment variable:

```bash
export NVIDIA_API_KEY="your-api-key-here"
# Or for OpenAI
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Example

Below we'll construct a simple Data Designer workflow that generates multilingual greetings.

```python
import os

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    InfoType,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
)

# Set your API key from build.nvidia.com
# Skip this step if you've already exported your key to the environemnt variable
os.environ["NVIDIA_API_KEY"] = "your-api-key-here"

# Create a DataDesigner instance
# This automatically configures the default model providers
data_designer = DataDesigner()

# Print out all the model providers available
data_designer.info.display(InfoType.MODEL_PROVIDERS)

# Create a config builder
# This automatically loads the default model configurations
config_builder = DataDesignerConfigBuilder()

# Print out all the model configurations available
config_builder.info.display(InfoType.MODEL_CONFIGS)

# Add a sampler column to randomly select a language
config_builder.add_column(
    SamplerColumnConfig(
        name="language",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["English", "Spanish", "French", "German", "Italian"],
        ),
    )
)

# Add an LLM text generation column
# We'll use the built-in 'nvidia-text' model alias
config_builder.add_column(
    LLMTextColumnConfig(
        name="greetings",
        model_alias="nvidia-text",
        prompt="""Write a casual and formal greeting in '{{language}}' language.""",
    )
)

# Run a preview to generate sample records
preview_results = data_designer.preview(config_builder=config_builder)

# Display a sample record
preview_results.display_sample_record()
```

🎉 Congratulations, you successfully ran one iteration designing your synthetic data. Follow along to learn more.

To learn more about the default providers and model configurations available, see the [Default Model Settings](concepts/models/default-model-settings.md) guide.
