# 🎨 NeMo Data Designer

[![GitHub](https://img.shields.io/badge/github-repo-952fc6?logo=github)](https://github.com/NVIDIA-NeMo/DataDesigner) [![License](https://img.shields.io/badge/License-Apache_2.0-0074df.svg)](https://opensource.org/licenses/Apache-2.0) [![NeMo Microservices](https://img.shields.io/badge/NeMo-Microservices-76b900)](https://docs.nvidia.com/nemo/microservices/latest/index.html)

👋 Welcome! Data Designer is an orchestration framework for generating high-quality synthetic data. You provide LLM endpoints (NVIDIA, OpenAI, vLLM, etc.), and Data Designer handles batching, parallelism, validation, and more.

**Configure** columns and models → **Preview** samples and iterate → **Create** your full dataset at scale.

Unlike raw LLM calls, Data Designer gives you statistical diversity, field correlations, automated validation, and reproducible workflows. For details, see [Architecture & Performance](concepts/architecture-and-performance.md).

📝 Want to hear from the team? Check out our **[Dev Notes](devnotes/index.md)** for deep dives, best practices, and insights.

## Install

```bash
pip install data-designer
```

## Setup

Get an API key from one of the default providers and set it as an environment variable:

```bash
# NVIDIA (build.nvidia.com) - recommended
export NVIDIA_API_KEY="your-api-key-here"

# OpenAI (platform.openai.com)
export OPENAI_API_KEY="your-openai-api-key-here"

# OpenRouter (openrouter.ai)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

Verify your configuration is ready:

```bash
data-designer config list
```

This displays the pre-configured model providers and models. See [CLI Configuration](concepts/models/configure-model-settings-with-the-cli.md) to customize.

## Your First Dataset

Let's generate multilingual greetings to see Data Designer in action:

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

# Initialize with default model providers
data_designer = DataDesigner()
config_builder = dd.DataDesignerConfigBuilder()

# Add a sampler column to randomly select a language
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="language",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["English", "Spanish", "French", "German", "Italian"],
        ),
    )
)

# Add an LLM text generation column
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="greeting",
        model_alias="nvidia-text",
        prompt="Write a casual and formal greeting in {{ language }}.",
    )
)

# Generate a preview
results = data_designer.preview(config_builder)
results.display_sample_record()
```

🎉 That's it! You've just designed your first synthetic dataset.

## 🚀 Next Steps

<div class="grid cards" markdown>

-   :material-book-open-variant: **[Tutorials](notebooks/README.md)**

    Step-by-step notebooks covering core features

-   :material-chef-hat: **[Recipes](recipes/cards.md)**

    Ready-to-use examples for common use cases

-   :material-cog: **[Concepts](concepts/columns.md)**

    Deep dive into columns, models, and configuration

</div>

## Learn More

- **[Deployment Options](concepts/deployment-options.md)** – Library vs. NeMo Microservice
- **[Model Configuration](concepts/models/default-model-settings.md)** – Configure LLM providers and models
- **[Architecture & Performance](concepts/architecture-and-performance.md)** – Optimize for throughput and scale
