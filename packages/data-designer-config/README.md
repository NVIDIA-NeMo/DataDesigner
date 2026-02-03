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
            model="nvidia/nemotron-3-nano-30b-a3b",
            inference_parameters=dd.ChatCompletionInferenceParams(temperature=0.7),
        ),
    ]
)

# Add columns
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="user_id",
        sampler_type=dd.SamplerType.UUID,
        params=dd.UUIDSamplerParams(prefix="user-"),
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

See main [README.md](https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/README.md) for more information.
