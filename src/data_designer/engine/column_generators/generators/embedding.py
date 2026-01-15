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

    def generate(self, data: dict) -> dict:
        deserialized_record = deserialize_json_values(data)
        input_texts = parse_list_string(deserialized_record[self.config.target_column])
        embeddings = self.model.generate_text_embeddings(input_texts=input_texts)
        data[self.config.name] = EmbeddingGenerationResult(embeddings=embeddings).model_dump(mode="json")
        return data
