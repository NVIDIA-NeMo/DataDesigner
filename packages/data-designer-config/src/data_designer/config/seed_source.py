# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import codecs
from abc import ABC
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from data_designer.config.utils.io_helpers import (
    RELATIVE_PATH_CWD_RESOLUTION_DESCRIPTION,
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_directory_path,
    validate_path_contains_files_of_type,
)

if TYPE_CHECKING:
    import pandas as pd


class SeedSource(BaseModel, ABC):
    """Base class for seed dataset configurations.

    All subclasses must define a `seed_type` field with a Literal value.
    This serves as a discriminated union discriminator.
    """

    seed_type: str


class LocalFileSeedSource(SeedSource):
    seed_type: Literal["local"] = "local"

    path: str = Field(
        ...,
        description=f"Path to a local seed dataset file or wildcard pattern. {RELATIVE_PATH_CWD_RESOLUTION_DESCRIPTION}",
    )

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


class FileSystemSeedSource(SeedSource, ABC):
    path: str = Field(
        ...,
        description=f"Directory containing seed artifacts. {RELATIVE_PATH_CWD_RESOLUTION_DESCRIPTION}",
    )
    file_pattern: str = Field(
        "*",
        description=(
            "Case-sensitive filename pattern used to match files under the provided directory. "
            "Patterns match basenames only, not relative paths."
        ),
    )
    recursive: bool = Field(
        True,
        description="Whether to search nested subdirectories under the provided directory for matching files.",
    )

    @field_validator("path", mode="after")
    def validate_path(cls, value: str) -> str:
        validate_directory_path(value)
        return value

    @field_validator("file_pattern", mode="after")
    def validate_file_pattern(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("🛑 FileSystemSeedSource.file_pattern must be a non-empty string.")
        if "/" in value or "\\" in value:
            raise ValueError("🛑 FileSystemSeedSource.file_pattern must match file names, not relative paths.")
        return value


class DirectorySeedSource(FileSystemSeedSource):
    seed_type: Literal["directory"] = "directory"


class FileContentsSeedSource(FileSystemSeedSource):
    seed_type: Literal["file_contents"] = "file_contents"

    encoding: str = Field(
        "utf-8",
        description="Text encoding used when reading matching files into the `content` column.",
    )

    @field_validator("encoding", mode="after")
    def validate_encoding(cls, value: str) -> str:
        try:
            codecs.lookup(value)
        except LookupError as error:
            raise ValueError(f"🛑 Unknown encoding: {value!r}. Use a valid Python codec name.") from error
        return value
