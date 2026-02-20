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
    """Filters batch rows based on a regex pattern."""

    def process_before_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        compiled = re.compile(self.config.pattern)
        mask = data[self.config.column].astype(str).apply(lambda v: bool(compiled.search(v)))
        if self.config.invert:
            mask = ~mask
        before = len(data)
        result = data[mask].reset_index(drop=True)
        logger.info(f"🔍 RegexFilter: {before} → {len(result)} rows (column={self.config.column!r})")
        return result
