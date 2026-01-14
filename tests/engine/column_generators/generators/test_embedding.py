# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from data_designer.config.column_configs import EmbeddingColumnConfig
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.column_generators.generators.embedding import (
    EmbeddingCellGenerator,
    EmbeddingGenerationResult,
)


@pytest.fixture
def stub_embedding_column_config():
    return EmbeddingColumnConfig(name="test_embedding", target_column="test_column", model_alias="test_model")


@pytest.fixture
def stub_embeddings() -> list[list[float]]:
    return [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_cell_generator_generation_strategy(
    stub_embedding_column_config: EmbeddingColumnConfig, stub_resource_provider: None
) -> None:
    generator = EmbeddingCellGenerator(config=stub_embedding_column_config, resource_provider=stub_resource_provider)
    assert generator.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


def test_embedding_cell_generator_generate(stub_embedding_column_config, stub_resource_provider, stub_embeddings):
    with patch.object(
        stub_resource_provider.model_registry.get_model.return_value,
        "generate_text_embeddings",
        return_value=stub_embeddings,
    ) as mock_generate:
        embedding_cell_generator = EmbeddingCellGenerator(
            config=stub_embedding_column_config, resource_provider=stub_resource_provider
        )
        data = embedding_cell_generator.generate(data={"test_column": "['test1', 'test2']"})
        assert stub_embedding_column_config.name in data
        assert data[stub_embedding_column_config.name] == EmbeddingGenerationResult(
            embeddings=stub_embeddings
        ).model_dump(mode="json")
        mock_generate.assert_called_once_with(input_texts=["test1", "test2"])
