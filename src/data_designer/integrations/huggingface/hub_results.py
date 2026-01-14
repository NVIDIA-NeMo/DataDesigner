# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults


@dataclass
class HubDatasetResults:
    """Results container for datasets pulled from Hugging Face Hub.

    This class contains the main dataset, analysis results, processor datasets,
    and processor artifacts that were pushed to the hub.
    """

    dataset: pd.DataFrame
    """The main dataset as a pandas DataFrame."""

    analysis: DatasetProfilerResults | None = None
    """Analysis results if available."""

    processor_datasets: dict[str, pd.DataFrame] | None = None
    """Dictionary of processor datasets, keyed by processor name."""

    processor_artifacts: dict[str, Path] | None = None
    """Dictionary of paths to processor artifacts, keyed by processor name."""

    metadata: dict[str, Any] | None = None
    """Metadata dictionary if available."""

    column_configs: list[dict[str, Any]] | None = None
    """Column configurations if available."""

    model_configs: list[dict[str, Any]] | None = None
    """Model configurations if available."""
