# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from data_designer.config.utils.io_helpers import (
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_path_contains_files_of_type,
)


class SeedSource(BaseModel, ABC):
    """Base class for seed dataset configurations.

    All subclasses must define a `seed_type` field with a Literal value.
    This serves as a discriminated union discriminator.
    """

    seed_type: str

    def get_column_names(self) -> list[str] | None:
        """Returns the column names from the seed dataset, or None if not available without I/O.

        Subclasses that can provide column names without expensive I/O (e.g., DataFrameSeedSource)
        should override this method. File-based sources return None by default, deferring
        column resolution to compile time.
        """
        return None


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

    def get_column_names(self) -> list[str]:
        """Returns column names by reading the file schema with DuckDB."""
        conn = duckdb.connect()
        describe_query = f"DESCRIBE SELECT * FROM '{self.path}'"
        column_descriptions = conn.execute(describe_query).fetchall()
        return [col[0] for col in column_descriptions]


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


class DataFrameSeedSource(SeedSource):
    seed_type: Literal["df"] = "df"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: SkipJsonSchema[pd.DataFrame] = Field(
        ...,
        exclude=True,
        description=(
            "DataFrame to use directly as the seed dataset. NOTE: if you need to write a Data Designer config, "
            "you must use `LocalFileSeedSource` instead, since DataFrame objects are not serializable."
        ),
    )

    def get_column_names(self) -> list[str]:
        """Returns column names from the DataFrame."""
        return list(self.df.columns)
