# data-designer-config

Configuration layer for NVIDIA DataDesigner synthetic data generation framework.

This package provides the configuration API for defining synthetic data generation pipelines. It's a lightweight dependency that can be used standalone for configuration management.

## Installation

```bash
pip install data-designer-config
```

## Features

- Configuration builder API (`DataDesignerConfigBuilder`)
- Column configuration types (Sampler, LLM, Expression, Validation, etc.)
- Model configurations with inference parameters
- Seed dataset configuration
- Constraint system for data generation
- Plugin system for extensibility

## Usage

```python
from data_designer.config import DataDesignerConfigBuilder, ModelConfig

# Create configuration builder
builder = DataDesignerConfigBuilder(
    model_configs=[
        ModelConfig(
            alias="my-model",
            model="meta/llama-3-70b-instruct",
            inference_parameters={"temperature": 0.7}
        )
    ]
)

# Add columns
builder.add_column(
    name="user_id",
    sampler_type="uuid",
    column_type="sampler"
)

builder.add_column(
    name="description",
    column_type="llm-text",
    prompt="Write a product description",
    model_alias="my-model"
)

# Build configuration
config = builder.build()
```

## Documentation

- [Full Documentation](https://nvidia.github.io/DataDesigner/)
- [Configuration Guide](https://nvidia.github.io/DataDesigner/configuration/)
- [API Reference](https://nvidia.github.io/DataDesigner/api/config/)

## License

Apache-2.0 - Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
