# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, computed_field

from data_designer.config.column_configs import EmbeddingColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel, GenerationStrategy
from data_designer.engine.processing.utils import deserialize_json_values, parse_list_string


class EmbeddingGenerationResult(BaseModel):
    embeddings: list[list[float]]

    @computed_field
    def num_embeddings(self) -> int:
        return len(self.embeddings)

    @computed_field
    def dimension(self) -> int:
        return len(self.embeddings[0]) if len(self.embeddings) > 0 else 0


class EmbeddingCellGenerator(ColumnGeneratorWithModel[EmbeddingColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def _prepare_embedding_inputs(self, data: dict) -> list[str]:
        cell_value = data[self.config.target_column]
        deserialized_value = deserialize_json_values(data)[self.config.target_column]
        if isinstance(deserialized_value, list):
            return [str(text).strip() for text in deserialized_value]
        if isinstance(deserialized_value, str):
            return parse_list_string(deserialized_value)
        if not isinstance(cell_value, str):
            raise ValueError(
                f"Embedding target column {self.config.target_column!r} must contain text or a list of texts, "
                f"got {cell_value!r}"
            )
        # A JSON scalar such as "123" or "true" is embedded as the user wrote it.
        return [cell_value.strip()]

    def generate(self, data: dict) -> dict:
        input_texts = self._prepare_embedding_inputs(data)
        embeddings = self.model.generate_text_embeddings(input_texts=input_texts)
        data[self.config.name] = EmbeddingGenerationResult(embeddings=embeddings).model_dump(mode="json")
        return data

    async def agenerate(self, data: dict) -> dict:
        """Native async generate using model.agenerate_text_embeddings."""
        input_texts = self._prepare_embedding_inputs(data)
        embeddings = await self.model.agenerate_text_embeddings(input_texts=input_texts)
        data[self.config.name] = EmbeddingGenerationResult(embeddings=embeddings).model_dump(mode="json")
        return data
