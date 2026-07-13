# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.io_helpers import list_processor_names, load_processor_dataset, read_parquet_dataset
from data_designer.config.utils.type_helpers import StrEnum, resolve_string_enum
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.engine.storage.media_storage import MediaStorage, StorageMode

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

BATCH_FILE_NAME_FORMAT = "batch_{batch_number:05d}.parquet"
SDG_CONFIG_FILENAME = "builder_config.json"
METADATA_FILENAME = "metadata.json"
FINAL_DATASET_FOLDER_NAME = "parquet-files"
PROCESSORS_OUTPUTS_FOLDER_NAME = "processors-files"
SELECTION_ACCEPTED_FOLDER_NAME = "selection-accepted"
SELECTION_CHECKPOINTS_FOLDER_NAME = "selection-checkpoints"
SELECTION_MEDIA_STAGING_FOLDER_NAME = "selection-media-staging"
SELECTION_PUBLICATION_STAGING_FOLDER_NAME = "selection-publication-staging"
SELECTION_SCHEMA_FILENAME = "schema.parquet"


class BatchStage(StrEnum):
    PARTIAL_RESULT = "partial_results_path"
    FINAL_RESULT = "final_dataset_path"
    DROPPED_COLUMNS = "dropped_columns_dataset_path"
    PROCESSORS_OUTPUTS = "processors_outputs_path"


class ResumeMode(StrEnum):
    NEVER = "never"
    ALWAYS = "always"
    IF_POSSIBLE = "if_possible"


class ArtifactStorage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifact_path: Path | str
    dataset_name: str = "dataset"
    final_dataset_folder_name: str = FINAL_DATASET_FOLDER_NAME
    partial_results_folder_name: str = "tmp-partial-parquet-files"
    dropped_columns_folder_name: str = "dropped-columns-parquet-files"
    processors_outputs_folder_name: str = PROCESSORS_OUTPUTS_FOLDER_NAME
    resume: ResumeMode = ResumeMode.NEVER
    _media_storage: MediaStorage = PrivateAttr(default=None)
    _metadata_defaults: dict[str, object] = PrivateAttr(default_factory=dict)

    @property
    def media_storage(self) -> MediaStorage:
        """Access media storage instance."""
        return self._media_storage

    @media_storage.setter
    def media_storage(self, value: MediaStorage) -> None:
        """Set media storage instance."""
        self._media_storage = value

    @property
    def artifact_path_exists(self) -> bool:
        return self.artifact_path.exists()

    @cached_property
    def resolved_dataset_name(self) -> str:
        dataset_path = self.artifact_path / self.dataset_name
        if dataset_path.exists() and len(list(dataset_path.iterdir())) > 0:
            if self.resume in (ResumeMode.ALWAYS, ResumeMode.IF_POSSIBLE):
                return self.dataset_name
            new_dataset_name = f"{self.dataset_name}_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}"
            logger.info(
                f"📂 Dataset path {str(dataset_path)!r} already exists. Dataset from this session"
                f"\n\t\t     will be saved to {str(self.artifact_path / new_dataset_name)!r} instead."
            )
            return new_dataset_name
        if self.resume == ResumeMode.ALWAYS:
            raise ArtifactStorageError(
                f"🛑 Cannot resume: no existing dataset found at {str(dataset_path)!r}. "
                "Run without resume=ResumeMode.ALWAYS to start a new generation."
            )
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
        return self.base_dataset_path / METADATA_FILENAME

    @property
    def partial_results_path(self) -> Path:
        return self.base_dataset_path / self.partial_results_folder_name

    @property
    def processors_outputs_path(self) -> Path:
        return self.base_dataset_path / self.processors_outputs_folder_name

    @property
    def selection_accepted_path(self) -> Path:
        return self.base_dataset_path / SELECTION_ACCEPTED_FOLDER_NAME

    @property
    def selection_checkpoints_path(self) -> Path:
        return self.base_dataset_path / SELECTION_CHECKPOINTS_FOLDER_NAME

    @property
    def selection_media_staging_path(self) -> Path:
        return self.base_dataset_path / SELECTION_MEDIA_STAGING_FOLDER_NAME

    @property
    def selection_publication_staging_path(self) -> Path:
        return self.base_dataset_path / SELECTION_PUBLICATION_STAGING_FOLDER_NAME

    @property
    def selection_schema_path(self) -> Path:
        return self.selection_accepted_path / SELECTION_SCHEMA_FILENAME

    @field_validator("artifact_path")
    def validate_artifact_path(cls, v: Path | str) -> Path:
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
            SELECTION_ACCEPTED_FOLDER_NAME,
            SELECTION_CHECKPOINTS_FOLDER_NAME,
            SELECTION_MEDIA_STAGING_FOLDER_NAME,
            SELECTION_PUBLICATION_STAGING_FOLDER_NAME,
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

        # Initialize media storage with DISK mode by default
        self._media_storage = MediaStorage(
            base_path=self.base_dataset_path,
            mode=StorageMode.DISK,
        )

        return self

    def set_media_storage_mode(self, mode: StorageMode) -> None:
        """Set media storage mode.

        Args:
            mode: StorageMode.DISK (save to disk) or StorageMode.DATAFRAME (store in memory)
        """
        self._media_storage.mode = mode

    def refresh_media_storage_path(self) -> None:
        """Re-point MediaStorage to the current base_dataset_path.

        Must be called after popping the resolved_dataset_name cache so that
        _media_storage.base_path and .images_dir reflect the updated directory.
        """
        images_subdir = self._media_storage.images_dir.name
        self._media_storage.base_path = self.base_dataset_path
        self._media_storage.images_dir = self.base_dataset_path / images_subdir

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

    def load_processor_dataset(self, processor_name: str) -> pd.DataFrame:
        """Load a processor's output dataset. Raises ArtifactStorageError if not found."""
        try:
            return load_processor_dataset(self.processors_outputs_path, processor_name)
        except FileNotFoundError as e:
            raise ArtifactStorageError(str(e)) from e

    def list_processor_names(self) -> list[str]:
        """Discover processor names from the processor outputs directory."""
        return list_processor_names(self.processors_outputs_path)

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
            df = lazy.pd.concat([df, df_dropped], axis=1)
        return df

    def clear_partial_results(self) -> None:
        """Remove any in-flight partial results left over from an interrupted run."""
        if self.partial_results_path.exists():
            shutil.rmtree(self.partial_results_path)

    def selection_partition_path(self, candidate_batch_id: int) -> Path:
        if candidate_batch_id < 0:
            raise ArtifactStorageError("🛑 Candidate batch number must be non-negative.")
        return self.selection_accepted_path / BATCH_FILE_NAME_FORMAT.format(batch_number=candidate_batch_id)

    def selection_checkpoint_path(self, candidate_batch_id: int) -> Path:
        if candidate_batch_id < 0:
            raise ArtifactStorageError("🛑 Candidate batch number must be non-negative.")
        return self.selection_checkpoints_path / f"batch_{candidate_batch_id:05d}.json"

    def write_selection_partition(self, candidate_batch_id: int, dataframe: pd.DataFrame) -> Path:
        """Atomically write one immutable accepted-row partition."""
        self.mkdir_if_needed(self.selection_accepted_path)
        path = self.selection_partition_path(candidate_batch_id)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            dataframe.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return path

    def write_selection_schema(self, dataframe: pd.DataFrame) -> Path:
        """Persist a zero-row schema anchor used to publish an empty partial result."""
        self.mkdir_if_needed(self.selection_accepted_path)
        tmp_path = self.selection_schema_path.with_name(f"{SELECTION_SCHEMA_FILENAME}.tmp.{os.getpid()}")
        try:
            dataframe.iloc[0:0].to_parquet(tmp_path, index=False)
            os.replace(tmp_path, self.selection_schema_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return self.selection_schema_path

    def write_selection_checkpoint(self, candidate_batch_id: int, marker: dict) -> Path:
        """Atomically commit a candidate batch marker."""
        self.mkdir_if_needed(self.selection_checkpoints_path)
        path = self.selection_checkpoint_path(candidate_batch_id)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(marker, file, indent=2, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return path

    def read_selection_checkpoints(self) -> list[dict]:
        if not self.selection_checkpoints_path.exists():
            return []
        markers: list[dict] = []
        for path in sorted(self.selection_checkpoints_path.glob("batch_*.json")):
            try:
                markers.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactStorageError(f"🛑 Selection checkpoint {path.name!r} is corrupt: {exc}") from exc
        return markers

    def clear_selection_transient_artifacts(self) -> None:
        """Discard in-flight selection and publication staging without touching committed work."""
        self.clear_partial_results()
        for path in (self.selection_media_staging_path, self.selection_publication_staging_path):
            if path.exists():
                shutil.rmtree(path)

    def clean_uncommitted_selection_batch(self, candidate_batch_id: int) -> None:
        """Remove deterministic artifacts for a candidate batch that has no committed marker."""
        self.selection_partition_path(candidate_batch_id).unlink(missing_ok=True)
        self.selection_checkpoint_path(candidate_batch_id).unlink(missing_ok=True)
        self._remove_candidate_batch_side_artifacts(candidate_batch_id)
        media_prefix = self.base_dataset_path / "images" / f"selection_batch_{candidate_batch_id:05d}"
        if media_prefix.exists():
            shutil.rmtree(media_prefix)
        staging = self.selection_media_staging_path / f"batch_{candidate_batch_id:05d}"
        if staging.exists():
            shutil.rmtree(staging)

    def begin_selection_media_batch(self, candidate_batch_id: int) -> None:
        """Route engine-managed media writes into candidate-scoped staging."""
        staging = self.selection_media_staging_path / f"batch_{candidate_batch_id:05d}"
        self.mkdir_if_needed(staging)
        self._media_storage.base_path = staging
        self._media_storage.images_dir = staging / self._media_storage.images_subdir

    def promote_selection_media(self, dataframe: pd.DataFrame, candidate_batch_id: int) -> pd.DataFrame:
        """Promote media referenced by accepted rows and rewrite their relative paths."""
        staging = self.selection_media_staging_path / f"batch_{candidate_batch_id:05d}"
        committed_relative_prefix = Path("images") / f"selection_batch_{candidate_batch_id:05d}"
        committed_prefix = self.base_dataset_path / committed_relative_prefix
        promoted_paths: dict[str, str] = {}

        def promote(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(f"{self._media_storage.images_subdir}/"):
                if value in promoted_paths:
                    return promoted_paths[value]
                source = staging / value
                try:
                    source.resolve().relative_to(staging.resolve())
                except ValueError:
                    return value
                if not source.is_file():
                    return value
                relative_tail = Path(value).relative_to(self._media_storage.images_subdir)
                destination = committed_prefix / relative_tail
                try:
                    destination.resolve().relative_to(committed_prefix.resolve())
                except ValueError:
                    return value
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, destination)
                promoted_path = (committed_relative_prefix / relative_tail).as_posix()
                promoted_paths[value] = promoted_path
                return promoted_path
            if isinstance(value, list):
                return [promote(item) for item in value]
            if isinstance(value, tuple):
                return tuple(promote(item) for item in value)
            if isinstance(value, dict):
                return {key: promote(item) for key, item in value.items()}
            return value

        promoted = dataframe.map(promote)
        self.finish_selection_media_batch(candidate_batch_id)
        return promoted

    def finish_selection_media_batch(self, candidate_batch_id: int) -> None:
        """Restore normal media storage and remove disposable candidate staging."""
        self._media_storage.base_path = self.base_dataset_path
        self._media_storage.images_dir = self.base_dataset_path / self._media_storage.images_subdir
        staging = self.selection_media_staging_path / f"batch_{candidate_batch_id:05d}"
        if staging.exists():
            shutil.rmtree(staging)

    def materialize_selection_dataset(self) -> Path:
        """Rebuild the published dataset from immutable accepted partitions."""
        if self.selection_publication_staging_path.exists():
            shutil.rmtree(self.selection_publication_staging_path)
        self.mkdir_if_needed(self.selection_publication_staging_path)
        partitions = sorted(self.selection_accepted_path.glob("batch_*.parquet"))
        if partitions:
            for partition in partitions:
                shutil.copy2(partition, self.selection_publication_staging_path / partition.name)
        elif self.selection_schema_path.is_file():
            shutil.copy2(
                self.selection_schema_path,
                self.selection_publication_staging_path / BATCH_FILE_NAME_FORMAT.format(batch_number=0),
            )
        else:
            raise ArtifactStorageError("🛑 Cannot publish record selection: no accepted partitions or schema found.")

        if self.final_dataset_path.exists():
            shutil.rmtree(self.final_dataset_path)
        os.replace(self.selection_publication_staging_path, self.final_dataset_path)
        return self.final_dataset_path

    def _remove_candidate_batch_side_artifacts(self, candidate_batch_id: int) -> None:
        filename = BATCH_FILE_NAME_FORMAT.format(batch_number=candidate_batch_id)
        (self.dropped_columns_dataset_path / filename).unlink(missing_ok=True)
        if self.processors_outputs_path.exists():
            for path in self.processors_outputs_path.rglob(filename):
                path.unlink(missing_ok=True)

    def move_partial_result_to_final_file_path(self, batch_number: int) -> Path:
        partial_result_path = self.create_batch_file_path(batch_number, batch_stage=BatchStage.PARTIAL_RESULT)
        if not partial_result_path.exists():
            raise ArtifactStorageError("🛑 Partial result file not found.")
        self.mkdir_if_needed(self._get_stage_path(BatchStage.FINAL_RESULT))
        final_file_path = self.create_batch_file_path(batch_number, batch_stage=BatchStage.FINAL_RESULT)
        shutil.move(partial_result_path, final_file_path)
        return final_file_path

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

    def get_parquet_file_paths(self) -> list[str]:
        """Get list of parquet file paths relative to base_dataset_path.

        Returns:
            List of relative paths to parquet files in the final dataset folder.
        """
        return [str(f.relative_to(self.base_dataset_path)) for f in sorted(self.final_dataset_path.glob("*.parquet"))]

    def get_processor_file_paths(self) -> dict[str, list[str]]:
        """Get processor output files organized by processor name.

        Returns:
            Dictionary mapping processor names to lists of relative file paths.
        """
        processor_files: dict[str, list[str]] = {}
        for name in self.list_processor_names():
            dir_path = self.processors_outputs_path / name
            file_path = self.processors_outputs_path / f"{name}.parquet"
            if dir_path.is_dir():
                processor_files[name] = [
                    str(f.relative_to(self.base_dataset_path)) for f in sorted(dir_path.rglob("*")) if f.is_file()
                ]
            elif file_path.is_file():
                processor_files[name] = [str(file_path.relative_to(self.base_dataset_path))]
        return processor_files

    def get_file_paths(self) -> dict[str, list[str] | dict[str, list[str]]]:
        """Get all file paths organized by type.

        Returns:
            Dictionary with 'parquet-files' and 'processor-files' keys.
        """
        file_paths = {
            "parquet-files": self.get_parquet_file_paths(),
        }
        processor_file_paths = self.get_processor_file_paths()
        if processor_file_paths:
            file_paths["processor-files"] = processor_file_paths

        return file_paths

    def read_metadata(self) -> dict:
        """Read metadata from the metadata.json file.

        Returns:
            Dictionary containing the metadata.

        Raises:
            FileNotFoundError: If metadata file doesn't exist.
        """
        with open(self.metadata_file_path, "r") as file:
            return json.load(file)

    def set_metadata_defaults(self, defaults: dict[str, object]) -> None:
        """Persist fields that should be included in every metadata write."""
        self._metadata_defaults.update(defaults)

    def write_metadata(self, metadata: dict) -> Path:
        """Write metadata to the metadata.json file.

        Args:
            metadata: Dictionary containing metadata to write.

        Returns:
            Path to the written metadata file.
        """
        self.mkdir_if_needed(self.base_dataset_path)
        metadata = {**self._metadata_defaults, **metadata}
        tmp_path = self.metadata_file_path.with_name(f"{self.metadata_file_path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp_path, "w") as file:
                json.dump(metadata, file, indent=2, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, self.metadata_file_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return self.metadata_file_path

    def update_metadata(self, updates: dict) -> Path:
        """Update existing metadata with new fields.

        Args:
            updates: Dictionary of fields to add/update in metadata.

        Returns:
            Path to the updated metadata file.
        """
        try:
            existing_metadata = self.read_metadata()
        except FileNotFoundError:
            existing_metadata = {}

        existing_metadata.update(updates)
        return self.write_metadata(existing_metadata)

    def _get_stage_path(self, stage: BatchStage) -> Path:
        return getattr(self, resolve_string_enum(stage, BatchStage).value)
