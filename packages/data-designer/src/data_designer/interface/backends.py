# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol

from data_designer.config.config_builder import DataDesignerConfigBuilder


class DataDesignerBackend(Protocol):
    """Execution backend hook for non-local Data Designer runtimes."""

    def create(
        self,
        *,
        data_designer: Any,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        input_dataset: Any | None = None,
    ) -> Any:
        """Create a dataset using the backend-specific data plane."""
