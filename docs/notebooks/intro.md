# Overview

Welcome to the Data Designer tutorial series! These hands-on notebooks will guide you through the core concepts and features of Data Designer, from basic synthetic data generation to advanced techniques like structured outputs and dataset seeding.

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

## 🚀 Setting Up Your Environment

### Local Setup Best Practices

To run these notebooks locally, we recommend using a virtual environment to manage dependencies:

=== "uv (Recommended)"

    ```bash
    # Create a new project directory
    mkdir data-designer-tutorials
    cd data-designer-tutorials

    # Initialize a new uv project
    uv init

    # Add data-designer and jupyter
    uv add data-designer jupyter

    # Launch Jupyter
    uv run jupyter notebook
    ```

=== "pip + venv"

    ```bash
    # Create a new project directory
    mkdir data-designer-tutorials
    cd data-designer-tutorials

    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install data-designer and jupyter
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

### Recommended Setup

For the best experience with Data Designer notebooks:

1. **Python Version:** Use Python 3.11 or later
2. **Virtual Environment:** Always use a virtual environment to avoid dependency conflicts
3. **Jupyter Extensions:** Consider installing JupyterLab for a more modern interface
4. **API Access:** Ensure you have valid API keys for your chosen LLM provider

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

## 💡 Tips for Success

- **Start Simple:** Begin with Tutorial 1 and work through the examples step by step
- **Experiment:** Modify the examples to generate your own datasets
- **Check Logs:** Data Designer provides detailed logging to help debug generation issues
- **Read Error Messages:** Error messages are designed to be helpful and actionable
- **Use the Registry:** Explore built-in samplers and generators using the registry system

## 🆘 Getting Help

If you run into issues:

1. Check the [GitHub Issues](https://github.com/NVIDIA-NeMo/DataDesigner/issues) for known problems
2. Review the [Contributing Guide](../CONTRIBUTING.md) if you'd like to report a bug or contribute
3. Consult the [Code Reference](../code_reference/column_configs.md) for detailed API documentation

---

Ready to get started? Head to [The Basics](1-the-basics.ipynb) to begin your journey! 🎨
