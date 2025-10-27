# 🎨 NeMo Data Designer
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Create synthetic datasets from scratch.

## Installation

```bash
git clone https://gitlab-master.nvidia.com/jogreco/data-designer.git
cd data-designer
uv sync
```

Test your installation:

```bash
make test
```

## Example Usage

```python
from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    InferenceParameters,
    LLMTextColumnConfig,
    ModelConfig,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
)

data_designer = DataDesigner(artifact_path="./artifacts")

# The model ID is from build.nvidia.com.
MODEL_ID = "nvidia/nvidia-nemotron-nano-9b-v2"

# We choose this alias to be descriptive for our use case.
MODEL_ALIAS = "nemotron-nano-v2"

# This sets reasoning to False for the nemotron-nano-v2 model.
SYSTEM_PROMPT = "/no_think"

model_configs = [
    ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        inference_parameters=InferenceParameters(
            temperature=0.5,
            top_p=1.0,
            max_tokens=1024,
        ),
    )
]

config_builder = DataDesignerConfigBuilder(model_configs=model_configs)


config_builder.add_column(
    SamplerColumnConfig(
        name="customer",
        sampler_type=SamplerType.PERSON,
        params=PersonSamplerParams(age_range=[18, 70]),
    )
)


config_builder.add_column(
    SamplerColumnConfig(
        name="product_category",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Electronics",
                "Clothing",
                "Home & Kitchen",
                "Books",
                "Home Office",
            ],
        ),
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="customer_review",
        prompt=(
            "You are a customer named {{ customer.first_name }} from {{ customer.city }}, "
            "{{ customer.state }}. Tell me about your experience working in the "
            "{{ product_category }} department of our company."
        ),        
        system_prompt=SYSTEM_PROMPT,
        model_alias=MODEL_ALIAS,
    )
)

preview = data_designer.preview(config_builder)

preview.display_sample_record()
```

## A note about about Person Sampling

> **Note:** The below usage is only temporary. The library's support for the Nemotron-Personas datasets will be evolve as we prepare to open source.

The PII and persona managed datasets have been updated for 25.11. If you want to use our Nemotron-Personas datasets for person / persona sampling, you need to do the following.

Download the datasets from NGC:
```bash
ngc registry resource download-version --org nvidian nvidian/nemo-llm/nemotron-personas-datasets:0.0.6-slim
```

The "slim" version is smaller for fast development. Remove the "-slim" to get the full datasets.

Tell `DataDesigner` where to find the datasets:
```python
data_designer = DataDesigner(artifact_path="./artifacts", blob_storage_path="/path/to/nemotron-personas-datasets")
```
