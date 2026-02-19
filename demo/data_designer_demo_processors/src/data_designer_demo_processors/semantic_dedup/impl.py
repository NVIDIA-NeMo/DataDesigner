from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from data_designer.engine.processing.processors.base import Processor
from data_designer_demo_processors.semantic_dedup.config import SemanticDedupProcessorConfig

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _suppress_transformers_logging() -> None:
    import transformers.utils.logging as tf_logging

    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()


class SemanticDedupProcessor(Processor[SemanticDedupProcessorConfig]):
    """Removes near-duplicate rows based on embedding cosine similarity."""

    def _initialize(self) -> None:
        _suppress_transformers_logging()
        self._model = SentenceTransformer(self.config.model_name)

    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        texts = data[self.config.column].astype(str).tolist()
        if len(texts) <= 1:
            return data

        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        sim_matrix = np.dot(embeddings, embeddings.T)

        keep = set(range(len(texts)))
        for i in range(len(texts)):
            if i not in keep:
                continue
            for j in range(i + 1, len(texts)):
                if j in keep and sim_matrix[i, j] >= self.config.similarity_threshold:
                    keep.discard(j)

        before = len(data)
        result = data.iloc[sorted(keep)].reset_index(drop=True)
        logger.info(f"🧹 SemanticDedup: {before} → {len(result)} rows (threshold={self.config.similarity_threshold})")
        return result
