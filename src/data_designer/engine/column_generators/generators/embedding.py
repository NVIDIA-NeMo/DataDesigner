# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from data_designer.config.column_configs import EmbeddingColumnConfig
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
    WithModelGeneration,
)
from data_designer.engine.processing.utils import deserialize_json_values, parse_list_string
from data_designer.engine.resources.resource_provider import ResourceType


class EmbeddingCellGenerator(WithModelGeneration, ColumnGenerator[EmbeddingColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="embedding_cell_generator",
            description="Generate embeddings for a text column.",
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
            required_resources=[ResourceType.MODEL_REGISTRY],
        )

    def generate(self, data: dict) -> dict:
        deserialized_record = deserialize_json_values(data)
        input_texts = parse_list_string(deserialized_record[self.config.target_column])
        embeddings = self.model.generate_text_embeddings(input_texts=input_texts)

        data[self.config.name] = {
            "embeddings": embeddings,
            "num_embeddings": len(embeddings),
            "dimension": len(embeddings[0]) if len(embeddings) > 0 else 0,
        }
        return data
