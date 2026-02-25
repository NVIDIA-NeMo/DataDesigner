# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from data_designer.engine.processing.processors.base import Processor
from data_designer_e2e_tests.plugins.regex_filter.config import RegexFilterProcessorConfig

if TYPE_CHECKING:
    import pandas as pd


class RegexFilterProcessor(Processor[RegexFilterProcessorConfig]):
    """Filters batch rows based on a regex pattern."""

    def process_before_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        compiled = re.compile(self.config.pattern)
        mask = data[self.config.column].astype(str).apply(lambda v: bool(compiled.search(v)))
        if self.config.invert:
            mask = ~mask
        return data[mask].reset_index(drop=True)
