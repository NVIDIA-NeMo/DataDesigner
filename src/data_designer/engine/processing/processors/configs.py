# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import field_validator

from data_designer.config.base import ConfigBase


class DropColumnsProcessorConfig(ConfigBase):
    column_names: list[str]
    dropped_column_parquet_file_name: str | None = None

    @field_validator("dropped_column_parquet_file_name")
    def validate_dropped_column_parquet_file_name(cls, v: str | None) -> str | None:
        if v is not None and not v.endswith(".parquet"):
            raise ValueError("Dropped column parquet file name must end with .parquet")
        return v
