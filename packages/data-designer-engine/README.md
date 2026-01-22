# data-designer-engine

Generation engine for NVIDIA DataDesigner synthetic data generation framework.

This package contains the execution engine that powers DataDesigner. It depends on `data-designer-config` and includes heavy dependencies like pandas, numpy, and LLM integration via litellm.

## Installation

```bash
pip install data-designer-engine
```

This automatically installs `data-designer-config` as a dependency.

## Features

- DAG-based dataset generation orchestration
- LLM integration via LiteLLM (supports 100+ providers)
- Sophisticated sampling generators (Person, Entity, etc.)
- Column validators (Python, SQL, Code, Remote)
- Dataset profiling and analysis
- Artifact storage and management

## Usage

```python
from data_designer.config import DataDesignerConfig
from data_designer.engine import compile_data_designer_config
from data_designer.engine.dataset_builders import ColumnWiseDatasetBuilder

# Assuming you have a config from data-designer-config
config = DataDesignerConfig(...)

# Compile configuration
compiled_config = compile_data_designer_config(config)

# Create builder and generate data
builder = ColumnWiseDatasetBuilder(compiled_config, resource_provider)
result = builder.build(num_records=100)
```

## Documentation

- [Full Documentation](https://nvidia.github.io/DataDesigner/)
- [Engine Architecture](https://nvidia.github.io/DataDesigner/architecture/)
- [API Reference](https://nvidia.github.io/DataDesigner/api/engine/)

## License

Apache-2.0 - Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
