# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn
from data_designer.lazy_heavy_imports import pd
from data_designer_e2e_tests.plugins.column_generator.config import DemoColumnGeneratorConfig

if TYPE_CHECKING:
    import pandas as pd


class DemoColumnGeneratorImpl(ColumnGeneratorFullColumn[DemoColumnGeneratorConfig]):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.config.name] = self.config.text.upper()

        return data
