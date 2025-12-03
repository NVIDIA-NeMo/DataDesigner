# 🎨 NeMo Data Designer Library

[![GitHub](https://img.shields.io/badge/github-repo-952fc6?logo=github)](https://github.com/NVIDIA-NeMo/DataDesigner) [![License](https://img.shields.io/badge/License-Apache_2.0-0074df.svg)](https://opensource.org/licenses/Apache-2.0) [![NeMo Microservices](https://img.shields.io/badge/NeMo-Microservices-76b900)](https://docs.nvidia.com/nemo/microservices/latest/index.html)

👋 Welcome to the Data Designer community! We're excited to have you here.

Data Designer is a **general framework** for generating **high-quality** synthetic data **from scratch** or using your own **seed data** as a starting point for domain-grounded data generation.

## Why Data Designer?

Generating high-quality synthetic data requires much more than iteratively calling an LLM.

Data Designer is **purpose-built** to support large-scale, high-quality data generation, including

  * **Diversity** – statistical distributions and variety that reflect real-world data patterns, not repetitive LLM outputs 
  * **Correlations** – meaningful relationships between fields that LLMs cannot maintain across independent calls
  * **Steerability** – flexible control over data characteristics throughout the generation process
  * **Validation** – automated quality checks and verification that data meets specifications
  * **Reproducibility** – shareable and reproducible generation workflows

## How does it work?

Data Designer helps you create datasets through an intuitive, **iterative** process:

1.  **⚙️ Configure** your model settings
    - Bring your own OpenAI-compatible model providers and models
    - Or use the default model providers and models to get started quickly
    - Learn more by reading the [model docs](concepts/models/default-model-settings.md)
2.  **🏗️ Design** your dataset
    - Iteratively design your dataset, column by column
    - Leverage tools like statistical samplers and LLMs to generate a variety of data types
    - Learn more by reading the [column docs](concepts/columns.md)

3.  **🔁 Preview** your results and iterate
    - Generate a preview dataset stored in memory for fast iteration
    - Inspect sample records and analysis results to refine your configuration
    - Try for yourself by running the [tutorial notebooks](notebooks/README.md)
4.  **🖼️ Create** your dataset
    - Generate your full dataset and save results to disk
    - Access the generated dataset and associated artifacts for downstream use
    - Give it a try by running the [tutorial notebooks](notebooks/README.md)

## Library and Microservice

Data Designer is available as both an open-source library and a NeMo microservice.

  * **Open-source Library**: Purpose-built for flexibility and customization, prioritizing UX excellence, modularity, and extensibility.
  * **NeMo Microservice**: An enterprise-grade solution that offers a seamless transition from the library, allowing you to leverage other NeMo microservices and generate datasets at scale. See the [microservice docs](https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/index.html) for more details.
