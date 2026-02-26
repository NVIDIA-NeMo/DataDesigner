# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import Field

from data_designer.config.base import ProcessorConfig


class RegexFilterProcessorConfig(ProcessorConfig):
    """Filters rows by regex pattern on a specified column."""

    processor_type: Literal["regex-filter"] = "regex-filter"
    column: str = Field(description="Column to match against.")
    pattern: str = Field(description="Regex pattern to match.")
    invert: bool = Field(default=False, description="If True, keep rows that do NOT match.")
