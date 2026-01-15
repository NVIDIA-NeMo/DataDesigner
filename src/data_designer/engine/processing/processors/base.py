# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

from data_designer.engine.configurable_task import ConfigurableTask, DataT, TaskConfigT


class Processor(ConfigurableTask[TaskConfigT], ABC):
    @abstractmethod
    def process(self, data: DataT, *, current_batch_number: int | None = None) -> DataT: ...
