# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import Field

from data_designer.config.processors import ProcessorConfig


class SemanticDedupProcessorConfig(ProcessorConfig):
    """Remove semantically similar rows using embeddings.

    This processor deduplicates the final dataset during the postprocess stage.
    """

    processor_type: Literal["semantic-dedup"] = "semantic-dedup"
    column: str = Field(description="Column to compute embeddings on for deduplication")
    similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold above which rows are considered duplicates",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name for computing embeddings",
    )
