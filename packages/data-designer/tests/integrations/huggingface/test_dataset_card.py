# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.integrations.huggingface.dataset_card import DataDesignerDatasetCard


@pytest.fixture
def stub_metadata() -> dict:
    """Stub metadata fixture with single column that can be used/modified by most tests."""
    return {
        "target_num_records": 100,
        "schema": {"col1": "string"},
        "column_statistics": [
            {
                "column_name": "col1",
                "num_records": 100,
                "num_unique": 100,
                "num_null": 0,
                "simple_dtype": "string",
                "column_type": "sampler",
            }
        ],
    }


def test_compute_size_category() -> None:
    """Test size category computation for various dataset sizes."""
    assert DataDesignerDatasetCard._compute_size_category(500) == "n<1K"
    assert DataDesignerDatasetCard._compute_size_category(5000) == "1K<n<10K"
    assert DataDesignerDatasetCard._compute_size_category(50000) == "10K<n<100K"
    assert DataDesignerDatasetCard._compute_size_category(500000) == "100K<n<1M"
    assert DataDesignerDatasetCard._compute_size_category(5000000) == "1M<n<10M"
    assert DataDesignerDatasetCard._compute_size_category(50000000) == "n>10M"


def test_from_metadata_minimal(stub_metadata: dict) -> None:
    """Test creating dataset card from minimal metadata."""
    # Add second column for this test
    stub_metadata["schema"]["col2"] = "int64"

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/dataset",
        description="Test dataset for unit testing.",
    )

    # Verify card was created
    assert card is not None
    assert "test/dataset" in str(card)
    assert "100" in str(card)
    assert "col1" in str(card)
    assert "2" in str(card)  # Number of columns


def test_from_metadata_with_sdg_config(stub_metadata: dict) -> None:
    """Test creating dataset card with sdg config."""
    # Customize for this test
    stub_metadata["target_num_records"] = 50
    stub_metadata["schema"] = {"name": "string", "age": "int64"}
    stub_metadata["column_statistics"] = [
        {
            "column_name": "name",
            "num_records": 50,
            "num_unique": 50,
            "num_null": 0,
            "simple_dtype": "string",
            "column_type": "sampler",
            "sampler_type": "person",
        },
        {
            "column_name": "age",
            "num_records": 50,
            "num_unique": 30,
            "num_null": 0,
            "simple_dtype": "int64",
            "column_type": "sampler",
            "sampler_type": "uniform",
        },
    ]

    sdg_config = {
        "data_designer": {
            "columns": [
                {"name": "name", "column_type": "sampler"},
                {"name": "age", "column_type": "sampler"},
            ]
        }
    }

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=sdg_config,
        repo_id="test/dataset-with-config",
        description="Test dataset with SDG config.",
    )

    # Verify card includes config info
    assert card is not None
    assert "sampler" in str(card)
    assert "2 column" in str(card)


def test_from_metadata_with_llm_columns(stub_metadata: dict) -> None:
    """Test creating dataset card with LLM column statistics."""
    # Customize for LLM test
    stub_metadata["target_num_records"] = 10
    stub_metadata["schema"] = {"prompt": "string", "response": "string"}
    stub_metadata["column_statistics"] = [
        {
            "column_name": "response",
            "num_records": 10,
            "num_unique": 10,
            "num_null": 0,
            "simple_dtype": "string",
            "column_type": "llm-text",
            "output_tokens_mean": 50.5,
            "input_tokens_mean": 20.3,
        }
    ]

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/llm-dataset",
        description="Test dataset with LLM columns.",
    )

    # Verify LLM statistics are included
    assert card is not None
    assert "Tokens:" in str(card) and "out" in str(card) and "in" in str(card)


def test_from_metadata_with_processors(stub_metadata: dict) -> None:
    """Test creating dataset card with processor outputs includes loading examples."""
    # Add processor files for this test
    stub_metadata["file_paths"] = {
        "parquet-files": ["parquet-files/batch_00000.parquet"],
        "processor-files": {
            "processor1": ["processors-files/processor1/batch_00000.parquet"],
            "processor2": ["processors-files/processor2/batch_00000.parquet"],
        },
    }

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/dataset-with-processors",
        description="Test dataset with processor outputs.",
    )

    card_str = str(card)
    assert card is not None
    assert "processor1" in card_str
    assert "processor2" in card_str
    assert '"processor1"' in card_str
    assert '"processor2"' in card_str
    assert "Load processor outputs" in card_str


def test_from_metadata_with_custom_description(stub_metadata: dict) -> None:
    """Test creating dataset card with custom description."""
    # Add second column for this test
    stub_metadata["schema"]["col2"] = "int64"

    description = "This dataset contains synthetic data for testing chatbot responses."

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/dataset-with-description",
        description=description,
    )

    card_str = str(card)
    assert card is not None
    assert "This dataset contains synthetic data for testing chatbot responses." in card_str


def test_from_metadata_description_placement(stub_metadata: dict) -> None:
    """Test that description appears in the correct location."""
    # Use 50 records for this test
    stub_metadata["target_num_records"] = 50
    stub_metadata["column_statistics"][0]["num_records"] = 50

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/dataset-description-placement",
        description="Test description placement.",
    )

    card_str = str(card)
    assert card is not None
    assert "Test description placement." in card_str
    assert "About NeMo Data Designer" in card_str
    # Description should appear before Dataset Summary
    desc_pos = card_str.find("Test description placement.")
    summary_pos = card_str.find("Dataset Summary")
    assert desc_pos < summary_pos


def test_from_metadata_default_tags(stub_metadata: dict) -> None:
    """Test that default tags are included when no custom tags are provided."""
    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/dataset-default-tags",
        description="Test dataset with default tags.",
    )

    card_str = str(card)
    assert card is not None
    # Check that default tags appear in the YAML frontmatter
    assert "- synthetic" in card_str
    assert "- datadesigner" in card_str


def test_from_metadata_with_custom_tags(stub_metadata: dict) -> None:
    """Test that custom tags are added to default tags."""
    custom_tags = ["chatbot", "conversation", "qa"]

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/dataset-custom-tags",
        description="Test dataset with custom tags.",
        tags=custom_tags,
    )

    card_str = str(card)
    assert card is not None
    # Check that both default and custom tags appear in the YAML frontmatter
    assert "- synthetic" in card_str
    assert "- datadesigner" in card_str
    assert "- chatbot" in card_str
    assert "- conversation" in card_str
    assert "- qa" in card_str


def test_from_metadata_tags_in_yaml_frontmatter(stub_metadata: dict) -> None:
    """Test that tags appear in the YAML frontmatter section."""
    # Use 50 records for this test
    stub_metadata["target_num_records"] = 50
    stub_metadata["column_statistics"][0]["num_records"] = 50

    card = DataDesignerDatasetCard.from_metadata(
        metadata=stub_metadata,
        sdg_config=None,
        repo_id="test/dataset-tags-frontmatter",
        description="Test dataset.",
        tags=["custom-tag"],
    )

    card_str = str(card)
    assert card is not None
    # Tags should appear before the main content (in YAML frontmatter)
    tags_section = card_str.find("tags:")
    quick_start_section = card_str.find("## 🚀 Quick Start")
    assert tags_section < quick_start_section
    assert tags_section != -1  # Make sure tags section exists
    # Verify tags appear before the closing of YAML frontmatter
    assert tags_section < card_str.find("---", tags_section)
