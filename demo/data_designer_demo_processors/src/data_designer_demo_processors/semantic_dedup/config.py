from __future__ import annotations

from typing import Literal

from pydantic import Field

from data_designer.config.processors import ProcessorConfig


class SemanticDedupProcessorConfig(ProcessorConfig):
    """Removes semantically similar rows using embedding similarity."""

    processor_type: Literal["semantic-dedup"] = "semantic-dedup"
    column: str = Field(description="Column to compute embeddings on.")
    similarity_threshold: float = Field(default=0.9, description="Cosine similarity threshold for deduplication.")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformers model name.")
