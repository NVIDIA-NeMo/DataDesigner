# 🎨 NeMo Data Designer Library

[![GitHub](https://img.shields.io/badge/github-repo-952fc6?logo=github)](https://github.com/NVIDIA-NeMo/DataDesigner) [![License](https://img.shields.io/badge/License-Apache_2.0-0074df.svg)](https://opensource.org/licenses/Apache-2.0) [![NeMo Microservices](https://img.shields.io/badge/NeMo-Microservices-76b900)](https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/index.html)

NeMo Data Designer is a **general framework for generating high-quality synthetic data**. It is use-case agnostic, modular, and extensible.

You can use it to generate entire datasets **from scratch** using a declarative config, or you can use **your own seed data** as a starting point for domain-grounded data generation.

## Why should I use Data Designer?

Generating high-quality synthetic data requires much more than iteratively calling an LLM. Simply making repeated LLM calls will produce data that lacks the structure and quality needed for real-world applications. Instead, effective synthetic data generation requires:

  * **Diversity** – statistical distributions and variety that reflect real-world data patterns, not repetitive LLM outputs 
  * **Correlations** – meaningful relationships between fields that LLMs cannot maintain across independent calls
  * **Steerability** – precise control over data characteristics through constraints and sampling strategies
  * **Validation** – automated quality checks and verification that data meets specifications
  * **Reproducibility** – deterministic generation with seed control for consistent, auditable results

Data Designer is a **purpose-built, modular, and maintained** framework that provides the missing features for large-scale, high-quality data generation, including:

  * **Statistical Samplers** and **Personas**
  * **Multi-step Workflows** and **Batch Jobs**
  * **External Validators**, **Code Execution**, and **Code Linting**
  * **Structured Output** and **Code Output**
  * **LLM Judge Scoring**
  * **Prompt Templating** using Jinja

## How does it work?

Data Designer helps you create datasets through a simple, iterative process:

1.  **⚙️ Set up** your model configs and providers.
2.  **🏗️ Design** your dataset, one column at a time.
3.  **🔁 Preview** your results and iterate fast.
4.  **🖼️ Create** your dataset at scale.

The core concept is designing your dataset using different **Column Types**. These columns can reference each other to build complex, multi-step generation workflows.

Key column types include:

  * **🌱 Seed**: Use data from your own seed dataset.
  * **🎲 Sampler**: Generate statistical data (e.g., categories, personas).
  * **📝 LLM Text**: Generate free-form text using an LLM.
  * **💻 LLM Code**: Generate code, with language specification for reliable extraction.
  * **🗂️ LLM Structured**: Generate structured JSON output.
  * **{{ 🧩 }} Expression**: Derive a column's value from others using Jinja templates.
  * **🔍 Validation**: Run quality checks (like code validation) on other columns.
  * **⚖️ LLM Judge**: Use an LLM to score or assess other generated data.

## Library and Microservice

Data Designer is available in two forms to support you from research to production:

  * **Open-source Library**: Purpose-built for researchers, prioritizing UX excellence, modularity, and extensibility.
  * **NeMo Microservice**: An enterprise-grade solution that offers a seamless transition from the library, allowing you to scale your data generation pipelines.

## Next Steps

[Installation](installation.md)
