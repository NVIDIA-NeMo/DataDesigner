# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.integrations.huggingface.dataset_card import DataDesignerDatasetCard


def test_compute_size_category() -> None:
    """Test size category computation for various dataset sizes."""
    assert DataDesignerDatasetCard._compute_size_category(500) == "n<1K"
    assert DataDesignerDatasetCard._compute_size_category(5000) == "1K<n<10K"
    assert DataDesignerDatasetCard._compute_size_category(50000) == "10K<n<100K"
    assert DataDesignerDatasetCard._compute_size_category(500000) == "100K<n<1M"
    assert DataDesignerDatasetCard._compute_size_category(5000000) == "1M<n<10M"
    assert DataDesignerDatasetCard._compute_size_category(50000000) == "n>10M"


def test_from_metadata_minimal() -> None:
    """Test creating dataset card from minimal metadata."""
    metadata = {
        "target_num_records": 100,
        "schema": {"col1": "string", "col2": "int64"},
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

    card = DataDesignerDatasetCard.from_metadata(
        metadata=metadata,
        sdg_config=None,
        repo_id="test/dataset",
    )

    # Verify card was created
    assert card is not None
    assert "test/dataset" in str(card)
    assert "100" in str(card)
    assert "col1" in str(card)
    assert "2" in str(card)  # Number of columns


def test_from_metadata_with_sdg_config() -> None:
    """Test creating dataset card with sdg config."""
    metadata = {
        "target_num_records": 50,
        "schema": {"name": "string", "age": "int64"},
        "column_statistics": [
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
        ],
    }

    sdg_config = {
        "data_designer": {
            "columns": [
                {"name": "name", "column_type": "sampler"},
                {"name": "age", "column_type": "sampler"},
            ]
        }
    }

    card = DataDesignerDatasetCard.from_metadata(
        metadata=metadata,
        sdg_config=sdg_config,
        repo_id="test/dataset-with-config",
    )

    # Verify card includes config info
    assert card is not None
    assert "sampler" in str(card)
    assert "2 column" in str(card)


def test_from_metadata_with_llm_columns() -> None:
    """Test creating dataset card with LLM column statistics."""
    metadata = {
        "target_num_records": 10,
        "schema": {"prompt": "string", "response": "string"},
        "column_statistics": [
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
        ],
    }

    card = DataDesignerDatasetCard.from_metadata(
        metadata=metadata,
        sdg_config=None,
        repo_id="test/llm-dataset",
    )

    # Verify LLM statistics are included
    assert card is not None
    assert "50.5" in str(card) or "Avg Output Tokens" in str(card)
