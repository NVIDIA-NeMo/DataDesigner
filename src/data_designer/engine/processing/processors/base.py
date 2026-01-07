# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata, DataT, TaskConfigT

if TYPE_CHECKING:
    from data_designer.engine.resources.resource_provider import ResourceType


class Processor(ConfigurableTask[TaskConfigT], ABC):
    @staticmethod
    def get_required_resources() -> list[ResourceType]:
        return []

    @staticmethod
    @abstractmethod
    def metadata() -> ConfigurableTaskMetadata: ...

    @abstractmethod
    def process(self, data: DataT, *, current_batch_number: int | None = None) -> DataT: ...
