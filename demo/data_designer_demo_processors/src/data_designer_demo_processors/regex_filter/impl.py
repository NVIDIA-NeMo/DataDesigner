# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor
from data_designer_demo_processors.regex_filter.config import RegexFilterProcessorConfig

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class RegexFilterProcessor(Processor[RegexFilterProcessorConfig]):
    """Filters rows based on regex matching on a specified column.

    Runs during preprocess to filter seed data before generation.
    """

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        column = self.config.column
        pattern = self.config.pattern
        invert = self.config.invert

        if column not in data.columns:
            logger.warning(f"⚠️ Column '{column}' not found in dataset. Skipping regex filter.")
            return data

        compiled = re.compile(pattern)
        mask = data[column].astype(str).apply(lambda x: bool(compiled.search(x)))

        if invert:
            mask = ~mask

        original_count = len(data)
        data = data[mask].reset_index(drop=True)
        filtered_count = original_count - len(data)

        action = "excluded" if not invert else "kept only non-matching"
        logger.info(f"🔍 Regex filter: {filtered_count} rows {action} (pattern: {pattern!r} on column '{column}')")

        return data
