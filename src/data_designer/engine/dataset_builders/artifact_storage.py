# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import shutil
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Union

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

from data_designer.config.utils.io_helpers import read_parquet_dataset
from data_designer.config.utils.type_helpers import StrEnum, resolve_string_enum
from data_designer.engine.dataset_builders.errors import ArtifactStorageError

logger = logging.getLogger(__name__)

BATCH_FILE_NAME_FORMAT = "batch_{batch_number:05d}.parquet"


class BatchStage(StrEnum):
    PARTIAL_RESULT = "partial_results_path"
    FINAL_RESULT = "final_dataset_path"
    DROPPED_COLUMNS = "dropped_columns_dataset_path"
    PROCESSORS_OUTPUTS = "processors_outputs_path"


class ArtifactStorage(BaseModel):
    artifact_path: Path | str
    dataset_name: str = "dataset"
    final_dataset_folder_name: str = "parquet-files"
    partial_results_folder_name: str = "tmp-partial-parquet-files"
    dropped_columns_folder_name: str = "dropped-columns-parquet-files"
    processors_outputs_folder_name: str = "processors-files"

    @property
    def artifact_path_exists(self) -> bool:
        return self.artifact_path.exists()

    @cached_property
    def resolved_dataset_name(self) -> str:
        dataset_path = self.artifact_path / self.dataset_name
        if dataset_path.exists() and len(list(dataset_path.iterdir())) > 0:
            new_dataset_name = f"{self.dataset_name}_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}"
            logger.info(
                f"📂 Dataset path {str(dataset_path)!r} already exists. Dataset from this session"
                f"\n\t\t     will be saved to {str(self.artifact_path / new_dataset_name)!r} instead."
            )
            return new_dataset_name
        return self.dataset_name

    @property
    def base_dataset_path(self) -> Path:
        return self.artifact_path / self.resolved_dataset_name

    @property
    def dropped_columns_dataset_path(self) -> Path:
        return self.base_dataset_path / self.dropped_columns_folder_name

    @property
    def final_dataset_path(self) -> Path:
        return self.base_dataset_path / self.final_dataset_folder_name

    @property
    def metadata_file_path(self) -> Path:
        return self.base_dataset_path / "metadata.json"

    @property
    def partial_results_path(self) -> Path:
        return self.base_dataset_path / self.partial_results_folder_name

    @property
    def processors_outputs_path(self) -> Path:
        return self.base_dataset_path / self.processors_outputs_folder_name

    @field_validator("artifact_path")
    def validate_artifact_path(cls, v: Union[Path, str]) -> Path:
        v = Path(v)
        if not v.is_dir():
            raise ArtifactStorageError("Artifact path must exist and be a directory")
        return v

    @model_validator(mode="after")
    def validate_folder_names(self):
        folder_names = [
            self.dataset_name,
            self.final_dataset_folder_name,
            self.partial_results_folder_name,
            self.dropped_columns_folder_name,
            self.processors_outputs_folder_name,
        ]

        for name in folder_names:
            if len(name) == 0:
                raise ArtifactStorageError("🛑 Directory names must be non-empty strings.")

        if len(set(folder_names)) != len(folder_names):
            raise ArtifactStorageError("🛑 Folder names must be unique (no collisions allowed).")

        invalid_chars = {"<", ">", ":", '"', "/", "\\", "|", "?", "*"}
        for name in folder_names:
            if any(char in invalid_chars for char in name):
                raise ArtifactStorageError(f"🛑 Directory name '{name}' contains invalid characters.")

        return self

    @staticmethod
    def mkdir_if_needed(path: Path | str) -> Path:
        """Create the directory if it does not exist."""
        path = Path(path)
        if not path.exists():
            logger.debug(f"📁 Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def read_parquet_files(path: Path) -> pd.DataFrame:
        return read_parquet_dataset(path)

    def create_batch_file_path(
        self,
        batch_number: int,
        batch_stage: BatchStage,
    ) -> Path:
        if batch_number < 0:
            raise ArtifactStorageError("🛑 Batch number must be non-negative.")
        return self._get_stage_path(batch_stage) / BATCH_FILE_NAME_FORMAT.format(batch_number=batch_number)

    def load_dataset(self, batch_stage: BatchStage = BatchStage.FINAL_RESULT) -> pd.DataFrame:
        return read_parquet_dataset(self._get_stage_path(batch_stage))

    def load_dataset_with_dropped_columns(self) -> pd.DataFrame:
        # The pyarrow backend has better support for nested data types.
        df = self.load_dataset()
        if (
            self.dropped_columns_dataset_path.exists()
            and self.create_batch_file_path(0, BatchStage.DROPPED_COLUMNS).is_file()
        ):
            logger.debug("Concatenating dropped columns to the final dataset.")
            df_dropped = self.load_dataset(batch_stage=BatchStage.DROPPED_COLUMNS)
            if len(df_dropped) != len(df):
                raise ArtifactStorageError(
                    "🛑 The dropped-columns dataset has a different number of rows than the main dataset. "
                    "Something unexpected must have happened to the dataset builder's artifacts."
                )
            # To ensure indexes are aligned and avoid silent misalignment (which would introduce NaNs),
            # check that the indexes are identical before concatenation.
            if not df.index.equals(df_dropped.index):
                raise ArtifactStorageError(
                    "🛑 The indexes of the main and dropped columns DataFrames are not aligned. "
                    "Something unexpected must have happened to the dataset builder's artifacts."
                )
            df = pd.concat([df, df_dropped], axis=1)
        return df

    def move_partial_result_to_final_file_path(self, batch_number: int) -> Path:
        partial_result_path = self.create_batch_file_path(batch_number, batch_stage=BatchStage.PARTIAL_RESULT)
        if not partial_result_path.exists():
            raise ArtifactStorageError("🛑 Partial result file not found.")
        self.mkdir_if_needed(self._get_stage_path(BatchStage.FINAL_RESULT))
        final_file_path = self.create_batch_file_path(batch_number, batch_stage=BatchStage.FINAL_RESULT)
        shutil.move(partial_result_path, final_file_path)
        return final_file_path

    def write_configs(self, json_file_name: str, configs: list[dict]) -> Path:
        self.mkdir_if_needed(self.base_dataset_path)
        with open(self.base_dataset_path / json_file_name, "w") as file:
            json.dump([c.model_dump(mode="json") for c in configs], file, indent=4)
        return self.base_dataset_path / json_file_name

    def write_batch_to_parquet_file(
        self,
        batch_number: int,
        dataframe: pd.DataFrame,
        batch_stage: BatchStage,
        subfolder: str | None = None,
    ) -> Path:
        file_path = self.create_batch_file_path(batch_number, batch_stage=batch_stage)
        self.write_parquet_file(file_path.name, dataframe, batch_stage, subfolder=subfolder)
        return file_path

    def write_parquet_file(
        self,
        parquet_file_name: str,
        dataframe: pd.DataFrame,
        batch_stage: BatchStage,
        subfolder: str | None = None,
    ) -> Path:
        subfolder = subfolder or ""
        self.mkdir_if_needed(self._get_stage_path(batch_stage) / subfolder)
        file_path = self._get_stage_path(batch_stage) / subfolder / parquet_file_name
        dataframe.to_parquet(file_path, index=False)
        return file_path

    def write_metadata(self, metadata: dict) -> Path:
        self.mkdir_if_needed(self.base_dataset_path)
        with open(self.metadata_file_path, "w") as file:
            json.dump(metadata, file)
        return self.metadata_file_path

    def _get_stage_path(self, stage: BatchStage) -> Path:
        return getattr(self, resolve_string_enum(stage, BatchStage).value)
