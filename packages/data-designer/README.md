# data-designer

Complete NVIDIA DataDesigner framework for synthetic data generation.

This is the full installation including the CLI, interface layer, and all dependencies. For lightweight installations, consider `data-designer-config` (config only) or `data-designer-engine` (config + engine).

## Installation

```bash
pip install data-designer
```

This installs all three packages:
- `data-designer-config` - Configuration layer
- `data-designer-engine` - Generation engine
- `data-designer` - CLI and interface

## Quick Start

```python
from data_designer import DataDesigner

# Initialize
dd = DataDesigner()

# Create configuration
builder = dd.create_config_builder(
    model_configs=[
        dd.create_model_config(
            alias="nvidia-text",
            model="meta/llama-3-70b-instruct"
        )
    ]
)

# Configure data generation
builder.add_column(name="id", sampler_type="uuid", column_type="sampler")
builder.add_column(
    name="review",
    column_type="llm-text",
    prompt="Write a product review",
    model_alias="nvidia-text"
)

# Generate data
result = dd.generate(config=builder.build(), num_records=100)
print(result.dataset)
```

## CLI Usage

```bash
# Interactive configuration
data-designer

# List available models
data-designer models list

# Configure providers
data-designer providers add nvidia --api-key YOUR_KEY
```

## Documentation

- [Full Documentation](https://nvidia.github.io/DataDesigner/)
- [Tutorials](https://nvidia.github.io/DataDesigner/tutorials/)
- [Recipes](https://nvidia.github.io/DataDesigner/recipes/)

## License

Apache-2.0 - Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
