# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata, DataT, TaskConfigT


class Processor(ConfigurableTask[TaskConfigT], ABC):
    @staticmethod
    @abstractmethod
    def metadata() -> ConfigurableTaskMetadata: ...

    @abstractmethod
    def process(self, data: DataT, *, current_batch_number: int | None = None) -> DataT: ...

    @staticmethod
    @abstractmethod
    def write_outputs_to_disk(
        processor_config: TaskConfigT, artifacts_path: Path, output_path: Path
    ) -> None: ...