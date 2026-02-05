# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC

from data_designer.engine.configurable_task import ConfigurableTask, DataT, TaskConfigT


class Processor(ConfigurableTask[TaskConfigT], ABC):
    """Base class for dataset processors.

    Processors transform data at different stages of the generation pipeline.
    Override the callback methods for the stages you want to handle.
    """

    def implements(self, method_name: str) -> bool:
        """Check if subclass overrides a callback method."""
        return getattr(type(self), method_name) is not getattr(Processor, method_name)

    def preprocess(self, data: DataT) -> DataT:
        """Called at PRE_GENERATION stage on seed data before batching.

        Override to filter or transform seed data before generation begins.

        Args:
            data: The full seed dataset.

        Returns:
            Transformed seed dataset.
        """
        return data

    def process_before_batch(self, data: DataT) -> DataT:
        """Called at PRE_BATCH stage before each batch is generated.

        Override to transform batch data before generation. Unlike preprocess,
        this operates on in-memory batch data without disk I/O.

        Args:
            data: The batch data before generation.

        Returns:
            Transformed batch data.
        """
        return data

    def process_after_batch(self, data: DataT, *, current_batch_number: int | None) -> DataT:
        """Called at POST_BATCH stage after each batch is generated.

        Override to process each batch of generated data.

        Args:
            data: The generated batch data.
            current_batch_number: The current batch number (0-indexed), or None in preview mode.

        Returns:
            Transformed batch data.
        """
        return data

    def postprocess(self, data: DataT) -> DataT:
        """Called at POST_GENERATION stage on the final combined dataset.

        Override to transform the complete generated dataset.

        Args:
            data: The final combined dataset.

        Returns:
            Transformed final dataset.
        """
        return data
