# 🎨 NeMo Data Designer

[![CI](https://github.com/NVIDIA-NeMo/DataDesigner/actions/workflows/ci.yml/badge.svg)](https://github.com/NVIDIA-NeMo/DataDesigner/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10 - 3.13](https://img.shields.io/badge/🐍_Python-3.10_|_3.11_|_3.12_|_3.13-blue.svg)](https://www.python.org/downloads/) [![NeMo Microservices](https://img.shields.io/badge/NeMo-Microservices-76b900)](https://docs.nvidia.com/nemo/microservices/latest/index.html) [![Code](https://img.shields.io/badge/Code-Documentation-8A2BE2.svg)](https://nvidia-nemo.github.io/DataDesigner/)

**Generate high-quality synthetic datasets from scratch or using your own seed data.**

---

## Welcome!

Data Designer helps you create synthetic datasets that go beyond simple LLM prompting. Whether you need diverse statistical distributions, meaningful correlations between fields, or validated high-quality outputs, Data Designer provides a flexible framework for building production-grade synthetic data.

## What can you do with Data Designer?

- **Generate diverse data** using statistical samplers, LLMs, or existing seed datasets
- **Control relationships** between fields with dependency-aware generation
- **Validate quality** with built-in Python, SQL, and custom local and remote validators
- **Score outputs** using LLM-as-a-judge for quality assessment
- **Iterate quickly** with preview mode before full-scale generation

---

## Quick Start

### 1. Install

```bash
pip install data-designer
```

Or install from source:

```bash
git clone https://github.com/NVIDIA-NeMo/DataDesigner.git
cd DataDesigner
make install
```

### 2. Set your API key

Get your API key from [build.nvidia.com](https://build.nvidia.com) or [OpenAI](https://platform.openai.com/api-keys):

```bash
export NVIDIA_API_KEY="your-api-key-here"
# Or use OpenAI
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Start generating data!
```python
from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
)

# Initialize with default settings
data_designer = DataDesigner()
config_builder = DataDesignerConfigBuilder()

# Add a product category
config_builder.add_column(
    SamplerColumnConfig(
        name="product_category",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Electronics", "Clothing", "Home & Kitchen", "Books"],
        ),
    )
)

# Generate personalized customer reviews
config_builder.add_column(
    LLMTextColumnConfig(
        name="review",
        model_alias="nvidia-text",
        prompt="""Write a brief product review for a {{ product_category }} item you recently purchased.""",
    )
)

# Preview your dataset
preview = data_designer.preview(config_builder=config_builder)
preview.display_sample_record()
```

---

## What's next?

### 📚 Learn more

- **[Quick Start Guide](https://nvidia-nemo.github.io/DataDesigner/latest/quick-start/)** – Detailed walkthrough with more examples
- **[Tutorial Notebooks](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/)** – Step-by-step interactive tutorials
- **[Column Types](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/columns/)** – Explore samplers, LLM columns, validators, and more
- **[Validators](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/validators/)** – Learn how to validate generated data with Python, SQL, and remote validators
- **[Model Configuration](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/model-configs/)** – Configure custom models and providers
- **[Person Sampling](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/person_sampling/)** – Learn how to sample realistic person data with demographic attributes

### 🔧 Configure models via CLI

```bash
data-designer config providers # Configure model providers
data-designer config models    # Set up your model configurations
data-designer config list      # View current settings
```

### 🤝 Get involved

- **[Contributing Guide](https://nvidia-nemo.github.io/DataDesigner/latest/CONTRIBUTING)** – Help improve Data Designer
- **[GitHub Issues](https://github.com/NVIDIA-NeMo/DataDesigner/issues)** – Report bugs or make a feature request

---

## License

Apache License 2.0 – see [LICENSE](LICENSE) for details.

---

## Citation

If you use NeMo Data Designer in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{nemo-data-designer,
  author = {The NeMo Data Designer Team},
  title = {NeMo Data Designer: A framework for generating synthetic data from scratch or based on your own seed data},
  howpublished = {\url{https://github.com/NVIDIA-NeMo/DataDesigner}},
  year = {2025},
  note = {GitHub Repository},
}
```
