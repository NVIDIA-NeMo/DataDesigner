# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import pandas as pd

from .models import ModelConfig, ModelProvider
from .utils.constants import DEFAULT_NUM_RECORDS
from .utils.info import InterfaceInfo

if TYPE_CHECKING:
    from .analysis.dataset_profiler import DatasetProfilerResults
    from .config_builder import DataDesignerConfigBuilder
    from .preview_results import PreviewResults


class ResultsProtocol(Protocol):
    def load_analysis(self) -> DatasetProfilerResults: ...
    def load_dataset(self) -> pd.DataFrame: ...


ResultsT = TypeVar("ResultsT", bound=ResultsProtocol)


class DataDesignerInterface(ABC, Generic[ResultsT]):
    @abstractmethod
    def create(
        self,
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int = DEFAULT_NUM_RECORDS,
    ) -> ResultsT: ...

    @abstractmethod
    def preview(
        self,
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int = DEFAULT_NUM_RECORDS,
    ) -> PreviewResults: ...

    @abstractmethod
    def get_default_model_configs(self) -> list[ModelConfig]: ...

    @abstractmethod
    def get_default_model_providers(self) -> list[ModelProvider]: ...

    @property
    @abstractmethod
    def info(self) -> InterfaceInfo: ...
