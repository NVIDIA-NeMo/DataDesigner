# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Hugging Face Hub push/pull functionality."""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
)
from data_designer.interface.huggingface import pull_from_hub
from data_designer.interface.results import DatasetCreationResults


@pytest.fixture
def stub_model_configs():
    """Mock model configs for testing."""
    from data_designer.config.models import InferenceParameters, ModelConfig

    return [
        ModelConfig(
            alias="nvidia-text",
            model="nvidia/nvidia-nemotron-nano-9b-v2",
            provider="nvidia",
            inference_parameters=InferenceParameters(
                temperature=0.5,
                top_p=1.0,
                max_tokens=1024,
            ),
        )
    ]


@pytest.fixture
def sample_dataset_config(stub_model_configs):
    """Create a sample dataset configuration matching the README example."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)

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

    # For integration tests, we'll mock LLM calls, but keep the config structure
    # Generate personalized customer reviews (will be mocked in tests)
    config_builder.add_column(
        LLMTextColumnConfig(
            name="review",
            model_alias="nvidia-text",
            prompt="""Write a brief product review for a {{ product_category }} item you recently purchased.""",
        )
    )

    return config_builder


@pytest.fixture
def simple_dataset_config():
    """Create a simple dataset configuration without LLM calls for faster testing."""
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

    # Add a simple numeric column
    from data_designer.config.sampler_params import UniformSamplerParams

    config_builder.add_column(
        SamplerColumnConfig(
            name="rating",
            sampler_type=SamplerType.UNIFORM,
            params=UniformSamplerParams(low=1, high=5),
        )
    )

    return config_builder


@pytest.mark.integration
@patch("data_designer.interface.huggingface.hub_mixin.Dataset")
@patch("data_designer.interface.huggingface.hub_mixin.HfApi")
@patch("data_designer.interface.huggingface.hub_mixin.load_dataset")
def test_push_and_pull_from_hub_integration(
    mock_load_dataset,
    mock_hf_api_class,
    mock_dataset_class,
    simple_dataset_config,
    tmp_path,
):
    """Integration test: create dataset, push to hub, pull from hub, verify round-trip."""
    # Initialize DataDesigner
    data_designer = DataDesigner()

    # Create a small dataset (10 records for testing) - using simple config without LLM
    num_records = 10
    results = data_designer.create(config_builder=simple_dataset_config, num_records=num_records)

    # Verify dataset was created
    original_df = results.load_dataset()
    assert len(original_df) == num_records
    assert "product_category" in original_df.columns
    assert "rating" in original_df.columns

    # Get original analysis
    original_analysis = results.load_analysis()

    # Mock Hugging Face Hub interactions
    mock_hf_dataset = MagicMock()
    mock_dataset_class.from_pandas.return_value = mock_hf_dataset

    mock_hf_api = MagicMock()
    mock_hf_api_class.return_value = mock_hf_api

    # Mock the uploaded files for pull_from_hub
    uploaded_files = {}

    def mock_upload_file(**kwargs):
        """Capture uploaded files."""
        path_or_fileobj = kwargs.get("path_or_fileobj")
        path_in_repo = kwargs.get("path_in_repo")
        if isinstance(path_or_fileobj, str):
            with open(path_or_fileobj, "rb") as f:
                uploaded_files[path_in_repo] = f.read()
        else:
            uploaded_files[path_in_repo] = path_or_fileobj.read()

    mock_hf_api.upload_file.side_effect = mock_upload_file

    # Mock load_dataset for pull_from_hub
    def mock_load_dataset_for_pull(repo_id, split=None, token=None, **kwargs):
        """Mock loading dataset from hub."""
        # Return the original dataset
        mock_hf_dataset_for_pull = MagicMock()
        mock_hf_dataset_for_pull.to_pandas.return_value = original_df
        return mock_hf_dataset_for_pull

    mock_load_dataset.side_effect = mock_load_dataset_for_pull

    # Mock hf_hub_download for pull_from_hub
    def mock_hf_hub_download(repo_id, filename, repo_type, token=None):
        """Mock downloading files from hub."""
        if filename in uploaded_files:
            # Create a temporary file with the uploaded content
            temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
            if filename.endswith(".json"):
                content = uploaded_files[filename].decode("utf-8") if isinstance(uploaded_files[filename], bytes) else uploaded_files[filename]
                temp_file.write(content)
                temp_file.close()
                return temp_file.name
        raise FileNotFoundError(f"File {filename} not found")

    # Mock list_repo_files
    def mock_list_repo_files(repo_id, repo_type, token=None):
        """Mock listing repo files."""
        return list(uploaded_files.keys())

    # Push to hub
    repo_id = "test-user/test-dataset"
    with patch("data_designer.interface.huggingface.hub_mixin.hf_hub_download", side_effect=mock_hf_hub_download), patch(
        "data_designer.interface.huggingface.hub_mixin.list_repo_files", side_effect=mock_list_repo_files
    ):
        results.push_to_hub(repo_id, token="test-token", generate_card=True)

        # Verify dataset was pushed
        mock_dataset_class.from_pandas.assert_called_once()
        mock_hf_dataset.push_to_hub.assert_called_once_with(repo_id, token="test-token")

        # Verify analysis.json was uploaded
        assert "analysis.json" in uploaded_files
        analysis_data = json.loads(uploaded_files["analysis.json"].decode("utf-8"))
        assert analysis_data["num_records"] == num_records

        # Verify README.md was uploaded
        assert "README.md" in uploaded_files
        readme_content = uploaded_files["README.md"].decode("utf-8")
        assert "NeMo Data Designer" in readme_content
        assert repo_id in readme_content

        # Pull from hub
        pulled_results = DatasetCreationResults.pull_from_hub(
            repo_id=repo_id,
            token="test-token",
            artifact_path=tmp_path / "pulled_artifacts",
        )

        # Verify pulled dataset matches original
        pulled_df = pulled_results.load_dataset()
        pd.testing.assert_frame_equal(pulled_df, original_df, check_dtype=False)

        # Verify pulled analysis matches original
        pulled_analysis = pulled_results.load_analysis()
        assert pulled_analysis.num_records == original_analysis.num_records
        assert pulled_analysis.target_num_records == original_analysis.target_num_records
        assert len(pulled_analysis.column_statistics) == len(original_analysis.column_statistics)

        # Verify config builder was reconstructed
        pulled_config_builder = pulled_results._config_builder
        assert pulled_config_builder is not None
        pulled_column_configs = pulled_config_builder.get_column_configs()
        assert len(pulled_column_configs) == 2  # product_category and rating

        # Verify artifact storage structure exists
        assert pulled_results.artifact_storage.base_dataset_path.exists()
        assert (pulled_results.artifact_storage.base_dataset_path / "parquet-files").exists()


@pytest.mark.integration
@patch("data_designer.interface.huggingface.hub_mixin.Dataset")
@patch("data_designer.interface.huggingface.hub_mixin.HfApi")
@patch("data_designer.interface.huggingface.hub_mixin.load_dataset")
def test_push_and_pull_with_pull_from_hub_function(
    mock_load_dataset,
    mock_hf_api_class,
    mock_dataset_class,
    simple_dataset_config,
):
    """Integration test: create dataset, push to hub, pull using pull_from_hub function."""
    # Initialize DataDesigner
    data_designer = DataDesigner()

    # Create a small dataset - using simple config without LLM
    num_records = 5
    results = data_designer.create(config_builder=simple_dataset_config, num_records=num_records)

    original_df = results.load_dataset()

    # Mock Hugging Face Hub interactions
    mock_hf_dataset = MagicMock()
    mock_dataset_class.from_pandas.return_value = mock_hf_dataset

    mock_hf_api = MagicMock()
    mock_hf_api_class.return_value = mock_hf_api

    uploaded_files = {}

    def mock_upload_file(**kwargs):
        """Capture uploaded files."""
        path_or_fileobj = kwargs.get("path_or_fileobj")
        path_in_repo = kwargs.get("path_in_repo")
        if isinstance(path_or_fileobj, str):
            with open(path_or_fileobj, "rb") as f:
                uploaded_files[path_in_repo] = f.read()
        else:
            uploaded_files[path_in_repo] = path_or_fileobj.read()

    mock_hf_api.upload_file.side_effect = mock_upload_file

    # Mock load_dataset for pull_from_hub
    def mock_load_dataset_for_pull(repo_id, split=None, token=None, **kwargs):
        """Mock loading dataset from hub."""
        mock_hf_dataset_for_pull = MagicMock()
        mock_hf_dataset_for_pull.to_pandas.return_value = original_df
        return mock_hf_dataset_for_pull

    mock_load_dataset.side_effect = mock_load_dataset_for_pull

    # Mock hf_hub_download for pull_from_hub
    def mock_hf_hub_download(repo_id, filename, repo_type, token=None):
        """Mock downloading files from hub."""
        if filename in uploaded_files:
            temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
            if filename.endswith(".json"):
                content = uploaded_files[filename].decode("utf-8") if isinstance(uploaded_files[filename], bytes) else uploaded_files[filename]
                temp_file.write(content)
                temp_file.close()
                return temp_file.name
        raise FileNotFoundError(f"File {filename} not found")

    # Mock list_repo_files
    def mock_list_repo_files(repo_id, repo_type, token=None):
        """Mock listing repo files."""
        return list(uploaded_files.keys())

    # Push to hub
    repo_id = "test-user/test-dataset-2"
    with patch("data_designer.interface.huggingface.hub_mixin.hf_hub_download", side_effect=mock_hf_hub_download), patch(
        "data_designer.interface.huggingface.hub_mixin.list_repo_files", side_effect=mock_list_repo_files
    ):
        results.push_to_hub(repo_id, token="test-token", generate_card=True)

        # Pull using pull_from_hub function
        hub_results = pull_from_hub(
            repo_id=repo_id,
            token="test-token",
            include_analysis=True,
            include_configs=True,
        )

        # Verify pulled dataset matches original
        pd.testing.assert_frame_equal(hub_results.dataset, original_df, check_dtype=False)

        # Verify analysis was loaded
        assert hub_results.analysis is not None
        assert hub_results.analysis.num_records == num_records

        # Verify configs were loaded
        assert hub_results.column_configs is not None
        assert len(hub_results.column_configs) == 2


@pytest.mark.integration
def test_real_push_to_hub(simple_dataset_config):
    """Real integration test: create dataset and push to actual Hugging Face Hub."""
    import os

    # Only run if HF_TOKEN is set
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN or HUGGINGFACE_HUB_TOKEN not set, skipping real push test")

    # Initialize DataDesigner
    data_designer = DataDesigner()

    # Create a small dataset - using simple config without LLM
    num_records = 1
    results = data_designer.create(config_builder=simple_dataset_config, num_records=num_records)

    # Verify dataset was created
    original_df = results.load_dataset()
    assert len(original_df) == num_records
    assert "product_category" in original_df.columns
    assert "rating" in original_df.columns

    # Push to actual Hugging Face Hub
    repo_id = "davidberenstein1957/datadesigner-test"
    print(f"\n🚀 Pushing dataset to {repo_id}...")
    results.push_to_hub(repo_id, token=token, generate_card=True)
    print(f"✅ Successfully pushed dataset to {repo_id}!")

