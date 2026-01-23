# data-designer-config

Configuration layer for NeMo Data Designer synthetic data generation framework.

This package provides the configuration API for defining synthetic data generation pipelines. It's a lightweight dependency that can be used standalone for configuration management.

## Installation

```bash
pip install data-designer-config
```

## Usage

```python
import data_designer.config as dd

# Initialize config builder with model config(s)
config_builder = dd.DataDesignerConfigBuilder(
    model_configs=[
        dd.ModelConfig(
            alias="my-model",
            model="meta/llama-3-70b-instruct",
            inference_parameters={"temperature": 0.7},
        ),
    ]
)

# Add columns
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="user_id",
        sampler_type=dd.SamplerType.UUID,
    )
)
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="description",
        prompt="Write a product description",
        model_alias="my-model",
    )
)

# Build configuration
config = config_builder.build()
```

See main [README.md](../../README.md) for more information.
