# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import transformers.utils.logging as transformers_logging
from sentence_transformers import SentenceTransformer

from data_designer.engine.processing.processors.base import Processor
from data_designer_demo_processors.semantic_dedup.config import SemanticDedupProcessorConfig

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SemanticDedupProcessor(Processor[SemanticDedupProcessorConfig]):
    """Removes semantically similar rows using embeddings.

    Runs during postprocess to deduplicate the final generated dataset.
    """

    def _initialize(self) -> None:
        # Suppress sentence-transformers/transformers logging noise
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()

        self._model = SentenceTransformer(self.config.model_name)

    def postprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        column = self.config.column
        threshold = self.config.similarity_threshold

        if column not in data.columns:
            logger.warning(f"⚠️ Column '{column}' not found in dataset. Skipping semantic dedup.")
            return data

        if len(data) == 0:
            return data

        texts = data[column].astype(str).tolist()
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms

        # Find duplicates using greedy approach: keep first occurrence, remove similar ones
        keep_indices = []
        for i in range(len(embeddings)):
            is_duplicate = False
            for kept_idx in keep_indices:
                similarity = np.dot(embeddings[i], embeddings[kept_idx])
                if similarity >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)

        original_count = len(data)
        data = data.iloc[keep_indices].reset_index(drop=True)
        removed_count = original_count - len(data)

        logger.info(
            f"🧹 Semantic dedup: removed {removed_count} similar rows (threshold: {threshold}, column: '{column}')"
        )

        return data
