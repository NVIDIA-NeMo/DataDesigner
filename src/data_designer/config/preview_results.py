# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import pandas as pd

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.visualization import WithRecordSamplerMixin


class PreviewResults(WithRecordSamplerMixin):
    def __init__(
        self,
        *,
        config_builder: DataDesignerConfigBuilder,
        dataset: Optional[pd.DataFrame] = None,
        analysis: Optional[DatasetProfilerResults] = None,
    ):
        """Creates a new instance with results from a Data Designer preview run.

        Args:
            config_builder: Data Designer configuration builder.
            dataset: Dataset of the preview run.
            analysis: Analysis of the preview run.
        """
        self.dataset: pd.DataFrame | None = dataset
        self.analysis: DatasetProfilerResults | None = analysis
        self._config_builder = config_builder
