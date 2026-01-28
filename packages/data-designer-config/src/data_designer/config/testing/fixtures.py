# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for config testing."""

from __future__ import annotations

import os
import tarfile
import tempfile
import textwrap
from typing import TYPE_CHECKING

import pytest
import yaml

from data_designer.config.analysis.column_statistics import GeneralColumnStatistics
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def stub_data_designer_config_str() -> str:
    return """
model_configs:
  - alias: my_own_code_model
    model: openai/meta/llama-3.3-70b-instruct
    inference_parameters:
      temperature:
        distribution_type: uniform
        params:
            low: 0.5
            high: 0.9
      top_p:
        distribution_type: manual
        params:
            values: [0.1, 0.2, 0.33]
            weights: [0.3, 0.2, 0.50]

seed_config:
  source:
    seed_type: hf
    path: datasets/test-repo/testing/data.csv
  sampling_strategy: shuffle

columns:
    - name: code_id
      sampler_type: uuid
      column_type: sampler
      params:
        prefix: code_
        short_form: true
        uppercase: true
    - name: age
      sampler_type: uniform
      column_type: sampler
      params:
        low: 35
        high: 88
    - name: domain
      sampler_type: category
      column_type: sampler
      params:
        values: [Healthcare, Finance, Education, Government]
    - name: topic
      sampler_type: category
      column_type: sampler
      params:
        values: [Web Development, Data Science, Machine Learning, Cloud Computing]
    - name: text
      column_type: llm-text
      prompt: Write a description of python code in topic {topic} and domain {domain}
      model_alias: my_own_code_model
    - name: code
      column_type: llm-code
      prompt: Write Python code that will be paired with the following prompt {text}
      model_alias: my_own_code_model
      code_lang: python
    - name: code_validation_result
      column_type: validation
      target_columns:
        - code
      validator_type: code
      validator_params:
        code_lang: python
    - name: code_judge_result
      model_alias: my_own_code_model
      column_type: llm-judge
      prompt: You are an expert in Python programming and make appropriate judgement on the quality of the code.
      scores:
        - name: Pythonic
          description: Pythonic Code and Best Practices (Does the code follow Python conventions and best practices?)
          options:
            "4": The code exemplifies Pythonic principles, making excellent use of Python-specific constructs, standard library modules and programming idioms; follows all relevant PEPs.
            "3": The code closely follows Python conventions and adheres to many best practices; good use of Python-specific constructs, standard library modules and programming idioms.
            "2": The code generally follows Python conventions but has room for better alignment with Pythonic practices.
            "1": The code loosely follows Python conventions, with several deviations from best practices.
            "0": The code does not follow Python conventions or best practices, using non-Pythonic approaches.
        - name: Readability
          description: Readability and Maintainability (Is the Python code easy to understand and maintain?)
          options:
            "4": The code is excellently formatted, follows PEP 8 guidelines, is elegantly concise and clear, uses meaningful variable names, ensuring high readability and ease of maintenance; organizes complex logic well. Docstrings are given in a Google Docstring format.
            "3": The code is well-formatted in the sense of code-as-documentation, making it relatively easy to understand and maintain; uses descriptive names and organizes logic clearly.
            "2": The code is somewhat readable with basic formatting and some comments, but improvements are needed; needs better use of descriptive names and organization.
            "1": The code has minimal formatting, making it hard to understand; lacks meaningful names and organization.
            "0": The code is unreadable, with no attempt at formatting or description.

constraints:
    - target_column: age
      operator: "lt"
      rhs: 65
"""


@pytest.fixture
def stub_data_designer_builder_config_str(stub_data_designer_config_str: str) -> str:
    return f"""
data_designer:
  {textwrap.indent(stub_data_designer_config_str, prefix="    ")}
"""


@pytest.fixture
def stub_data_designer_config(stub_data_designer_config_str: str) -> DataDesignerConfig:
    json_config = yaml.safe_load(stub_data_designer_config_str)
    return DataDesignerConfig.model_validate(json_config)


@pytest.fixture
def stub_model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            alias="stub-model",
            model="stub-model",
            inference_parameters=ChatCompletionInferenceParams(
                temperature=0.9,
                top_p=0.9,
                max_tokens=2048,
            ),
        )
    ]


@pytest.fixture
def stub_model_providers() -> list[ModelProvider]:
    return [
        ModelProvider(
            name="provider-1",
            endpoint="https://api.provider-1.com/v1",
            api_key="PROVIDER_1_API_KEY",
        )
    ]


@pytest.fixture
def stub_empty_builder(stub_model_configs: list[ModelConfig]) -> DataDesignerConfigBuilder:
    """Test builder with model configs."""
    return DataDesignerConfigBuilder(model_configs=stub_model_configs)


@pytest.fixture
def stub_complete_builder(stub_data_designer_builder_config_str: str) -> DataDesignerConfigBuilder:
    return DataDesignerConfigBuilder.from_config(config=stub_data_designer_builder_config_str)


@pytest.fixture
def stub_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["John", "Jane", "Jim", "Jill", "Mike", "Mary", "Mark", "Martha", "Alex", "Alice", "Bob", "Bella"],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 22, 28, 65, 38],
            "city": [
                "New York",
                "Los Angeles",
                "Chicago",
                "Houston",
                "Miami",
                "Seattle",
                "San Francisco",
                "Boston",
                "Denver",
                "Austin",
                "Portland",
                "Atlanta",
            ],
            "state": ["NY", "CA", "IL", "TX", "FL", "WA", "CA", "MA", "CO", "TX", "OR", "GA"],
            "zip": [
                "10001",
                "90001",
                "60601",
                "77001",
                "33101",
                "98101",
                "94101",
                "02101",
                "80201",
                "73301",
                "97201",
                "30301",
            ],
            "email": [
                "john@example.com",
                "jane@example.com",
                "jim@example.com",
                "jill@example.com",
                "mike@example.com",
                "mary@example.com",
                "mark@example.com",
                "martha@example.com",
                "alex.smith@example.co.uk",
                "alice.wu@example.ca",
                "bob.martin@example.org",
                "bella.rossi@example.it",
            ],
            "phone": [
                "123-456-7890",
                "213-555-1234",
                "312-222-3333",
                "713-444-5555",
                "305-888-9999",
                "206-777-8888",
                "415-999-0000",
                "617-111-2222",
                "+44 20 7946 0958",
                "+1-416-555-0199",
                "+39 06 6982 1234",
                "+49 30 123456",
            ],
            "address": [
                "123 Main St",
                "456 Oak Ave",
                "789 Pine Rd",
                "101 Maple Blvd",
                "202 Elm St",
                "303 Cedar Ave",
                "404 Spruce Dr",
                "505 Birch Ln",
                "12 Baker St",
                "88 King St W",
                "Via Roma 1",
                "Unter den Linden 5",
            ],
        }
    )


@pytest.fixture
def stub_dataset_tar_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid parquet files with actual data
        df1 = pd.DataFrame({"id": ["1", "2"], "name": ["test", "sample"]})
        df2 = pd.DataFrame({"id": ["3", "4"], "name": ["data", "example"]})

        # Write parquet files
        os.makedirs(temp_dir + "/dataset", exist_ok=True)
        df1.to_parquet(temp_dir + "/dataset/dataset-001.parquet", index=False)
        df2.to_parquet(temp_dir + "/dataset/dataset-002.parquet", index=False)

        # Create tar file
        tar_path = temp_dir + "/dataset.tar"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(temp_dir + "/dataset/dataset-001.parquet", arcname="dataset/dataset-001.parquet")
            tar.add(temp_dir + "/dataset/dataset-002.parquet", arcname="dataset/dataset-002.parquet")
        with open(tar_path, "rb") as tar_file:
            yield tar_file


@pytest.fixture
def stub_dataset_profiler_results() -> DatasetProfilerResults:
    stub_column_statistics = GeneralColumnStatistics(
        column_name="some",
        num_records=1,
        num_unique=1,
        num_null=0,
        pyarrow_dtype="string",
        simple_dtype="string",
    )
    return DatasetProfilerResults(
        num_records=1,
        target_num_records=100,
        column_statistics=[stub_column_statistics],
        side_effect_column_names=None,
        column_profiles=None,
    )


@pytest.fixture
def stub_sampler_only_config_builder(stub_model_configs: list[ModelConfig]) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(
            name="uuid", sampler_type="uuid", params={"prefix": "code_", "short_form": True, "uppercase": True}
        )
    )
    config_builder.add_column(
        SamplerColumnConfig(name="category", sampler_type="category", params={"values": ["a", "b", "c"]})
    )
    config_builder.add_column(
        SamplerColumnConfig(name="uniform", sampler_type="uniform", params={"low": 1, "high": 100})
    )
    return config_builder
