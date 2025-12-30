# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_designer.config.utils.io_helpers import (
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_path_contains_files_of_type,
)


class SeedDatasetConfig(BaseModel):
    """Base class for seed dataset configurations.

    All subclasses must define a `seed_type` field with a Literal value.
    This serves as a discriminated union discriminator.
    """

    seed_type: str


class LocalFileSeedConfig(SeedDatasetConfig):
    seed_type: Literal["local"] = "local"

    path: str

    @field_validator("path", mode="after")
    def validate_path(cls, v: str) -> str:
        valid_wild_card_versions = {f"*{ext}" for ext in VALID_DATASET_FILE_EXTENSIONS}
        if any(v.endswith(wildcard) for wildcard in valid_wild_card_versions):
            parts = v.split("*.")
            file_path = parts[0]
            file_extension = parts[-1]
            validate_path_contains_files_of_type(file_path, file_extension)
        else:
            validate_dataset_file_path(v)
        return v


class HuggingFaceSeedConfig(SeedDatasetConfig):
    seed_type: Literal["hf"] = "hf"

    dataset: str = Field(pattern=r"^hf://datasets/*")
    token: str | None = None


class DataFrameSeedConfig(SeedDatasetConfig):
    seed_type: Literal["df"] = "df"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame = Field(exclude=True, default_factory=lambda: pd.DataFrame())


SeedDatasetConfigT = Annotated[
    LocalFileSeedConfig | HuggingFaceSeedConfig | DataFrameSeedConfig,
    Field(discriminator="seed_type"),
]
