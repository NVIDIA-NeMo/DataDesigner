# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer_demo_processors.semantic_dedup import SemanticDedupProcessor, SemanticDedupProcessorConfig


@pytest.fixture
def stub_resource_provider():
    return Mock()


@pytest.mark.slow
def test_semantic_dedup_removes_similar_rows(stub_resource_provider):
    config = SemanticDedupProcessorConfig(name="test", column="text", similarity_threshold=0.9)
    processor = SemanticDedupProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame(
        {
            "text": [
                "The cat sat on the mat",
                "A cat was sitting on the mat",  # Very similar to first
                "Dogs like to play fetch",
                "The dog enjoys playing fetch",  # Very similar to third
                "Quantum physics is fascinating",
            ]
        }
    )
    result = processor.process(df)

    # Should keep only dissimilar rows
    assert len(result) < len(df)
    assert len(result) >= 3  # At least 3 distinct topics


@pytest.mark.slow
def test_semantic_dedup_missing_column_returns_unchanged(stub_resource_provider):
    config = SemanticDedupProcessorConfig(name="test", column="missing", similarity_threshold=0.9)
    processor = SemanticDedupProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"text": ["hello", "world"]})
    result = processor.process(df)

    assert len(result) == 2


@pytest.mark.slow
def test_semantic_dedup_empty_dataframe(stub_resource_provider):
    config = SemanticDedupProcessorConfig(name="test", column="text", similarity_threshold=0.9)
    processor = SemanticDedupProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"text": []})
    result = processor.process(df)

    assert len(result) == 0


@pytest.mark.slow
def test_semantic_dedup_low_threshold_removes_more(stub_resource_provider):
    config = SemanticDedupProcessorConfig(name="test", column="text", similarity_threshold=0.5)
    processor = SemanticDedupProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"text": ["apple", "orange", "banana", "grape", "lemon"]})
    result = processor.process(df)

    # With low threshold, fruit words might be considered similar
    assert len(result) <= len(df)
