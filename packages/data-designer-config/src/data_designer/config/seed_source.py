# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args

from pydantic import Field, field_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.errors import InvalidFilePathError
from data_designer.config.utils.io_helpers import (
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_path_contains_files_of_type,
)
from data_designer.plugin_manager import PluginManager

if TYPE_CHECKING:
    import pandas as pd


class SeedSource(ConfigBase, ABC):
    """Base class for seed dataset configurations.

    All subclasses must define a `seed_type` field with a Literal value.
    This serves as a discriminated union discriminator.
    """

    seed_type: str


class LocalFileSeedSource(SeedSource):
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

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, path: str) -> Self:
        df.to_parquet(path, index=False)
        return cls(path=path)


class HuggingFaceSeedSource(SeedSource):
    seed_type: Literal["hf"] = "hf"

    path: str = Field(
        ...,
        description=(
            "Path to the seed data in HuggingFace. Wildcards are allowed. Examples include "
            "'datasets/my-username/my-dataset/data/000_00000.parquet', 'datasets/my-username/my-dataset/data/*.parquet', "
            "and 'datasets/my-username/my-dataset/**/*.parquet'"
        ),
    )
    token: str | None = None
    endpoint: str = "https://huggingface.co"


class DirectorySeedTransform(ConfigBase, ABC):
    """Base class for full-batch directory seed transforms."""

    transform_type: str


class DirectoryListingTransform(DirectorySeedTransform):
    transform_type: Literal["directory_listing"] = "directory_listing"


plugin_manager = PluginManager()


def build_directory_seed_transform_type() -> Any:
    directory_seed_transform_type = plugin_manager.inject_into_directory_transform_type_union(DirectoryListingTransform)
    if get_args(directory_seed_transform_type):
        return Annotated[directory_seed_transform_type, Field(discriminator="transform_type")]
    return directory_seed_transform_type


DirectorySeedTransformT = build_directory_seed_transform_type()


class DirectorySeedSource(SeedSource):
    seed_type: Literal["directory"] = "directory"

    path: str = Field(..., description="Directory containing seed artifacts.")
    file_pattern: str = Field("*", description="Filename pattern used to match files under the provided directory.")
    recursive: bool = Field(
        True,
        description="Whether to search nested subdirectories under the provided directory for matching files.",
    )
    transform: DirectorySeedTransformT | None = Field(
        default=None,
        description="Optional full-batch transform applied to the matched files before seeding.",
    )

    @field_validator("path", mode="after")
    def validate_path(cls, value: str) -> str:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise InvalidFilePathError(f"🛑 Path {path} is not a directory.")
        return str(path)

    @field_validator("file_pattern", mode="after")
    def validate_file_pattern(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("🛑 DirectorySeedSource.file_pattern must be a non-empty string.")
        if "/" in value or "\\" in value:
            raise ValueError("🛑 DirectorySeedSource.file_pattern must match file names, not relative paths.")
        return value
