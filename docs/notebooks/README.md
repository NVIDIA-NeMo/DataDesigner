# Overview

Welcome to the Data Designer tutorial series! These hands-on notebooks will guide you through the core concepts and features of Data Designer, from basic synthetic data generation to advanced techniques like structured outputs and dataset seeding.

## 🚀 Setting Up Your Environment

### Local Setup Best Practices

First, download the tutorial [from the release assets](https://github.com/NVIDIA-NeMo/DataDesigner/releases/latest/download/data_designer_tutorial.zip).
To run the tutorial notebooks locally, we recommend using a virtual environment to manage dependencies:

=== "uv (Recommended)"

    ```bash
    # Extract tutorial notebooks
    unzip data_designer_tutorial.zip
    cd data_designer_tutorial

    # Launch Jupyter
    uv run jupyter notebook
    ```

=== "pip + venv"

    ```bash
    # Extract tutorial notebooks
    unzip data_designer_tutorial.zip
    cd data_designer_tutorial

    # Create Python virtual environment and install required packages
    python -m venv venv
    source venv/bin/activate
    pip install data-designer jupyter

    # Launch Jupyter
    jupyter notebook
    ```

### API Keys and Authentication

Data Designer is able to interface with various LLM providers. You'll need to set up API keys for the models you want to use:

```bash
# For NVIDIA API Catalog (build.nvidia.com)
export NVIDIA_API_KEY="your-api-key-here"

# For OpenAI
export OPENAI_API_KEY="your-api-key-here"
```

For more information, check the [Quick Start](../quick-start.md), [Default Model Settings](../models/default-model-settings.md) and how to [Configure Model Settings Using The CLI](../models/configure-model-settings-with-the-cli.md).

## 📚 Tutorial Series

The tutorials are designed to be completed in sequence, building upon concepts introduced in previous notebooks:

### [1. The Basics](1-the-basics.ipynb)

Learn the fundamentals of Data Designer by generating a simple product review dataset. This notebook covers:

- Setting up the `DataDesigner` interface
- Configuring models and inference parameters
- Using built-in samplers (Category, Person, Uniform)
- Generating LLM text columns with dependencies
- Understanding the generation workflow

**Start here if you're new to Data Designer!**

### [2. Structured Outputs and Jinja Expressions](2-structured-outputs-and-jinja-expressions.ipynb)

Explore more advanced data generation capabilities:

- Creating structured JSON outputs with schemas
- Using Jinja expressions for derived columns
- Combining samplers with structured data
- Building complex data dependencies
- Working with nested data structures

### [3. Seeding with an External Dataset](3-seeding-with-a-dataset.ipynb)

Learn how to leverage existing datasets to guide synthetic data generation:

- Loading and using seed datasets
- Sampling from real data distributions
- Combining seed data with LLM generation
- Creating realistic synthetic data based on existing patterns

## 📖 Important Documentation Sections

Before diving into the tutorials, familiarize yourself with these key documentation sections:

### Getting Started

- **[Installation](../installation.md)** - Detailed installation instructions for various setups
- **[Welcome Guide](../index.md)** - Overview of Data Designer capabilities and architecture

### Core Concepts

Understanding these concepts will help you make the most of the tutorials:

- **[Columns](../concepts/columns.md)** - Learn about different column types (Sampler, LLM, Expression, Validation, etc.)
- **[Validators](../concepts/validators.md)** - Understand how to validate generated data with Python, SQL, and remote validators
- **[Plugins](../concepts/plugins.md)** - Extend Data Designer with custom functionality

### Code Reference

Quick reference guides for the main configuration objects:

- **[column_configs](../code_reference/column_configs.md)** - All column configuration types
- **[config_builder](../code_reference/config_builder.md)** - The `DataDesignerConfigBuilder` API
- **[data_designer_config](../code_reference/data_designer_config.md)** - Main configuration schema
- **[validator_params](../code_reference/validator_params.md)** - Validator configuration options
