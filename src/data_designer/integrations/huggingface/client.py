# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import DatasetCardData, HfApi, get_token, hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.column_types import DataDesignerColumnType, get_column_display_order
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.engine.analysis.utils.column_statistics_calculations import (
    convert_pyarrow_dtype_to_simple_dtype,
)
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.integrations.huggingface.dataset_card import DataDesignerDatasetCard
from data_designer.integrations.huggingface.hub_results import HubDatasetResults

logger = logging.getLogger(__name__)


class HasDataset(Protocol):
    """Protocol for classes that have a load_dataset method."""

    def load_dataset(self) -> pd.DataFrame: ...


class HasArtifactStorage(Protocol):
    """Protocol for classes that have artifact_storage with metadata_file_path."""

    @property
    def artifact_storage(self) -> Any: ...


def resolve_hf_token(token: str | None) -> str | None:
    """Resolve the Hugging Face token from parameter or huggingface_hub.

    This function tries to resolve a token in the following order:
    1. Token provided as parameter
    2. huggingface_hub's get_token() (checks environment variables, cache, config file, etc.)

    Args:
        token: Token provided as parameter.

    Returns:
        Resolved token or None if not found.
    """
    if token is not None:
        return token

    try:
        token = get_token()
        if token:
            return token
    except Exception:
        pass

    return None


def parse_size_category(num_records: int) -> str:
    """Parse dataset size into Hugging Face size category.

    Uses the same category names as Argilla's size_categories_parser.

    Args:
        num_records: Number of records in the dataset.

    Returns:
        Size category string matching Hugging Face format (e.g., "n<1K", "1K<n<10K", etc.).
    """
    size_categories = {
        1_000: "n<1K",
        10_000: "1K<n<10K",
        100_000: "10K<n<100K",
        1_000_000: "100K<n<1M",
        10_000_000: "1M<n<10M",
        100_000_000: "10M<n<100M",
        1_000_000_000: "100M<n<1B",
        10_000_000_000: "1B<n<10B",
        100_000_000_000: "10B<n<100B",
        1_000_000_000_000: "100B<n<1T",
    }

    for size, category in size_categories.items():
        if num_records < size:
            return category
    return "n>1T"


def pydantic_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model to a dict, handling enum fields properly.

    Args:
        obj: Pydantic model instance.

    Returns:
        Dictionary representation of the object.
    """
    if not hasattr(obj, "model_dump"):
        return obj

    result = obj.model_dump(mode="json")
    for key in ["column_type", "sampler_type"]:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, dict) and "value" in value:
            continue
        if hasattr(value, "value"):
            result[key] = {"value": value.value}
        elif isinstance(value, str):
            result[key] = {"value": value}
    return result


class HuggingFaceHubClient:
    """Client for pushing and pulling datasets to/from Hugging Face Hub.

    This class encapsulates all Hugging Face Hub operations and can be composed
    into other classes to provide hub functionality without using mixins.
    """

    def __init__(
        self,
        dataset_provider: HasDataset,
        artifact_storage_provider: HasArtifactStorage | None = None,
        analysis: DatasetProfilerResults | None = None,
        config_builder: DataDesignerConfigBuilder | None = None,
    ) -> None:
        """Initialize the Hugging Face Hub client.

        Args:
            dataset_provider: Object that provides the dataset via load_dataset().
            artifact_storage_provider: Object that provides artifact storage.
            analysis: Optional analysis results for dataset card generation.
            config_builder: Optional config builder for dataset card generation.
        """
        self._dataset_provider = dataset_provider
        self._artifact_storage_provider = artifact_storage_provider
        self._analysis = analysis
        self._config_builder = config_builder

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        generate_card: bool = True,
        **kwargs: Any,
    ) -> None:
        """Push the dataset to Hugging Face Hub.

        This method converts the pandas DataFrame to a HuggingFace Dataset, pushes it to
        the Hugging Face Hub, and optionally generates and uploads a dataset card.

        Args:
            repo_id: The ID of the Hugging Face Hub repository (e.g., "username/dataset-name").
            token: Hugging Face token for authentication. If None, will check environment
                variables HF_TOKEN or HUGGINGFACE_HUB_TOKEN.
            generate_card: Whether to generate and upload a dataset card. Defaults to True.
            **kwargs: Additional arguments to pass to `dataset.push_to_hub()`.

        Raises:
            ArtifactStorageError: If there's an error loading the dataset or metadata.
        """
        resolved_token = resolve_hf_token(token)
        dataset_df = self._dataset_provider.load_dataset()
        hf_dataset = Dataset.from_pandas(dataset_df)
        hf_dataset.push_to_hub(repo_id, token=resolved_token, **kwargs)

        if self._artifact_storage_provider:
            self._upload_additional_artifacts(repo_id, resolved_token)

        if generate_card:
            self._upload_dataset_card(repo_id, resolved_token, dataset_df)

    def _upload_additional_artifacts(
        self,
        repo_id: str,
        token: str | None,
    ) -> None:
        """Upload additional artifacts to Hugging Face Hub.

        This includes:
        - Analysis results (as JSON)
        - Processor datasets (as parquet files)
        - Processor artifacts (directories)
        - Configuration files (column_configs.json, model_configs.json)

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.
        """
        if not self._artifact_storage_provider:
            return

        hf_api = HfApi(token=token)
        artifact_storage = self._artifact_storage_provider.artifact_storage

        self._upload_analysis(hf_api, repo_id)
        self._upload_processor_artifacts(hf_api, repo_id, artifact_storage)
        self._upload_metadata(hf_api, repo_id, artifact_storage)
        self._upload_config_files(hf_api, repo_id, artifact_storage)

    def _upload_analysis(self, hf_api: HfApi, repo_id: str) -> None:
        """Upload analysis results as JSON.

        Args:
            hf_api: Hugging Face API client.
            repo_id: The ID of the Hugging Face Hub repository.
        """
        if self._analysis is None:
            return

        try:
            analysis_json = self._analysis.model_dump(mode="json")
            with TemporaryDirectory() as tmpdir:
                analysis_path = Path(tmpdir) / "analysis.json"
                with open(analysis_path, "w") as f:
                    json.dump(analysis_json, f, indent=2, default=str)
                hf_api.upload_file(
                    path_or_fileobj=str(analysis_path),
                    path_in_repo="analysis.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
        except Exception as e:
            logger.warning(f"Failed to upload analysis results: {e}")

    def _upload_processor_artifacts(
        self,
        hf_api: HfApi,
        repo_id: str,
        artifact_storage: Any,
    ) -> None:
        """Upload processor datasets and artifacts.

        Args:
            hf_api: Hugging Face API client.
            repo_id: The ID of the Hugging Face Hub repository.
            artifact_storage: Artifact storage object.
        """
        if not hasattr(artifact_storage, "processors_outputs_path"):
            return

        processors_path = artifact_storage.processors_outputs_path
        if not processors_path.exists():
            return

        for processor_dir in processors_path.iterdir():
            if not processor_dir.is_dir():
                continue
            processor_name = processor_dir.name
            self._upload_processor_dataset(hf_api, repo_id, processor_dir, processor_name)
            self._upload_processor_files(hf_api, repo_id, processors_path, processor_dir, processor_name)

    def _upload_processor_dataset(
        self,
        hf_api: HfApi,
        repo_id: str,
        processor_dir: Path,
        processor_name: str,
    ) -> None:
        """Upload a processor dataset as a parquet file.

        Args:
            hf_api: Hugging Face API client.
            repo_id: The ID of the Hugging Face Hub repository.
            processor_dir: Directory containing the processor files.
            processor_name: Name of the processor.
        """
        parquet_files = list(processor_dir.glob("*.parquet"))
        if not parquet_files:
            return

        try:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            combined_df = pd.concat(dfs, ignore_index=True)

            with TemporaryDirectory() as tmpdir:
                processor_parquet = Path(tmpdir) / f"{processor_name}.parquet"
                combined_df.to_parquet(processor_parquet, index=False)
                hf_api.upload_file(
                    path_or_fileobj=str(processor_parquet),
                    path_in_repo=f"processors/{processor_name}.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
        except Exception as e:
            logger.warning(f"Failed to upload processor dataset {processor_name}: {e}")

    def _upload_processor_files(
        self,
        hf_api: HfApi,
        repo_id: str,
        processors_path: Path,
        processor_dir: Path,
        processor_name: str,
    ) -> None:
        """Upload non-parquet files from a processor directory.

        Args:
            hf_api: Hugging Face API client.
            repo_id: The ID of the Hugging Face Hub repository.
            processors_path: Base path for all processors.
            processor_dir: Directory containing the processor files.
            processor_name: Name of the processor.
        """
        for artifact_file in processor_dir.rglob("*"):
            if not artifact_file.is_file() or artifact_file.suffix == ".parquet":
                continue
            try:
                relative_path = artifact_file.relative_to(processors_path)
                hf_api.upload_file(
                    path_or_fileobj=str(artifact_file),
                    path_in_repo=f"processors/{relative_path.as_posix()}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            except Exception as e:
                logger.warning(f"Failed to upload processor artifact {artifact_file}: {e}")

    def _upload_metadata(
        self,
        hf_api: HfApi,
        repo_id: str,
        artifact_storage: Any,
    ) -> None:
        """Upload metadata file with sanitized file paths.

        Args:
            hf_api: Hugging Face API client.
            repo_id: The ID of the Hugging Face Hub repository.
            artifact_storage: Artifact storage object.
        """
        if not hasattr(artifact_storage, "metadata_file_path"):
            return

        metadata_path = artifact_storage.metadata_file_path
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            sanitized_metadata = self._sanitize_metadata_file_paths(metadata, artifact_storage)

            with TemporaryDirectory() as tmpdir:
                sanitized_metadata_path = Path(tmpdir) / "metadata.json"
                with open(sanitized_metadata_path, "w") as f:
                    json.dump(sanitized_metadata, f, indent=2, default=str)
                hf_api.upload_file(
                    path_or_fileobj=str(sanitized_metadata_path),
                    path_in_repo="metadata.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
        except Exception as e:
            logger.warning(f"Failed to upload metadata: {e}")

    def _sanitize_metadata_file_paths(self, metadata: dict[str, Any], artifact_storage: Any) -> dict[str, Any]:
        """Sanitize file paths in metadata by converting local paths to remote paths.

        Args:
            metadata: Metadata dictionary that may contain file_paths.
            artifact_storage: Artifact storage object.

        Returns:
            Metadata dictionary with sanitized file paths.
        """
        if "file_paths" not in metadata or not isinstance(metadata["file_paths"], list):
            return metadata

        sanitized_paths = []
        base_path = artifact_storage.base_dataset_path

        for file_path in metadata["file_paths"]:
            path_obj = Path(str(file_path))
            sanitized = None

            if path_obj.is_absolute():
                try:
                    relative_path = path_obj.relative_to(base_path)
                    sanitized = f"data/{relative_path.as_posix()}"
                except ValueError:
                    pass

            if not sanitized:
                path_str = str(file_path)
                if "parquet-files" in path_str:
                    idx = path_str.find("parquet-files")
                    remaining = path_str[idx + len("parquet-files") :].lstrip("/\\") if idx != -1 else path_obj.name
                    sanitized = f"data/parquet-files/{remaining}"
                else:
                    sanitized = f"data/{path_obj.name}"

            sanitized_paths.append(sanitized)

        result = metadata.copy()
        if sanitized_paths:
            result["file_paths"] = sanitized_paths
        else:
            result.pop("file_paths", None)
        return result

    def _upload_config_files(
        self,
        hf_api: HfApi,
        repo_id: str,
        artifact_storage: Any,
    ) -> None:
        """Upload configuration files (column_configs.json, model_configs.json).

        Args:
            hf_api: Hugging Face API client.
            repo_id: The ID of the Hugging Face Hub repository.
            artifact_storage: Artifact storage object.
        """
        if not hasattr(artifact_storage, "base_dataset_path"):
            return

        base_path = artifact_storage.base_dataset_path
        config_files = ["column_configs.json", "model_configs.json"]
        for config_file in config_files:
            config_path = base_path / config_file
            if config_path.exists():
                try:
                    hf_api.upload_file(
                        path_or_fileobj=str(config_path),
                        path_in_repo=config_file,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                except Exception as e:
                    logger.warning(f"Failed to upload {config_file}: {e}")

    def _upload_dataset_card(
        self,
        repo_id: str,
        token: str | None,
        dataset_df: pd.DataFrame,
    ) -> None:
        """Generate and upload the dataset card to Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.
            dataset_df: The dataset as a pandas DataFrame.

        Raises:
            ArtifactStorageError: If analysis or config_builder is missing.
        """
        if self._analysis is None or self._config_builder is None:
            raise ArtifactStorageError(
                "Cannot generate dataset card: missing analysis or config_builder. "
                "Ensure the client was initialized with analysis and config_builder."
            )

        metadata = self._load_metadata_for_card()
        template_variables = self._build_card_template_variables(
            dataset_df=dataset_df,
            analysis=self._analysis,
            config_builder=self._config_builder,
            metadata=metadata,
            repo_id=repo_id,
        )

        card = self._create_dataset_card(dataset_df, template_variables)
        self._save_and_upload_card(card, repo_id, token)

    def _load_metadata_for_card(self) -> dict[str, Any] | None:
        """Load and sanitize metadata for dataset card generation.

        Returns:
            Sanitized metadata dictionary or None if not available.
        """
        if not self._artifact_storage_provider:
            return None

        artifact_storage = self._artifact_storage_provider.artifact_storage
        if not hasattr(artifact_storage, "metadata_file_path"):
            return None

        metadata_path = artifact_storage.metadata_file_path
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return self._sanitize_metadata_file_paths(metadata, artifact_storage)
        except Exception:
            return None

    def _build_card_template_variables(
        self,
        dataset_df: pd.DataFrame,
        analysis: DatasetProfilerResults,
        config_builder: DataDesignerConfigBuilder,
        metadata: dict[str, Any] | None,
        repo_id: str,
    ) -> dict[str, Any]:
        """Build template variables for the dataset card.

        Args:
            dataset_df: The dataset as a pandas DataFrame.
            analysis: Profiling analysis results.
            config_builder: Configuration builder.
            metadata: Optional metadata dictionary.
            repo_id: Repository ID.

        Returns:
            Dictionary of template variables.
        """
        column_configs = config_builder.get_column_configs()
        column_names = set(dataset_df.columns)

        all_columns = self._build_column_info(dataset_df, column_names)
        unconfigured_columns = self._find_unconfigured_columns(dataset_df, column_names, column_configs)
        sample_records = self._build_sample_records(dataset_df)
        config_types = self._build_config_types_summary(column_configs)
        column_stats_by_type = self._build_column_stats_by_type(analysis)

        return {
            "size_categories": parse_size_category(len(dataset_df)),
            "num_records": len(dataset_df),
            "target_num_records": analysis.target_num_records,
            "percent_complete": analysis.percent_complete,
            "num_columns": len(dataset_df.columns),
            "repo_id": repo_id,
            "metadata": metadata or {},
            "column_configs": [pydantic_to_dict(col_config) for col_config in column_configs] if column_configs else [],
            "unconfigured_columns": unconfigured_columns,
            "all_columns": all_columns,
            "column_statistics": (
                [pydantic_to_dict(stat) for stat in analysis.column_statistics] if analysis.column_statistics else []
            ),
            "column_stats_by_type": column_stats_by_type,
            "sorted_column_types": self._sort_column_types(column_stats_by_type),
            "num_samples": len(sample_records),
            "sample_records": sample_records,
            "config_types": config_types,
        }

    def _build_column_info(self, dataset_df: pd.DataFrame, column_names: set[str]) -> dict[str, str]:
        """Build column information dictionary with normalized types.

        Args:
            dataset_df: The dataset as a pandas DataFrame.
            column_names: Set of column names.

        Returns:
            Dictionary mapping column names to their normalized types.
        """
        all_columns: dict[str, str] = {}
        for col_name in sorted(column_names):
            try:
                normalized_type = convert_pyarrow_dtype_to_simple_dtype(dataset_df[col_name].dtype.pyarrow_dtype)
            except Exception:
                normalized_type = str(dataset_df[col_name].dtype)
            all_columns[col_name] = normalized_type
        return all_columns

    def _find_unconfigured_columns(
        self,
        dataset_df: pd.DataFrame,
        column_names: set[str],
        column_configs: list[Any] | None,
    ) -> dict[str, str]:
        """Find columns that don't have configurations.

        Args:
            dataset_df: The dataset as a pandas DataFrame.
            column_names: Set of all column names.
            column_configs: List of column configurations.

        Returns:
            Dictionary mapping unconfigured column names to their types.
        """
        if not column_configs:
            return {}

        configured_names = {col.name for col in column_configs}
        unconfigured = column_names - configured_names
        return {col_name: str(dataset_df[col_name].dtype) for col_name in sorted(unconfigured)}

    def _build_sample_records(self, dataset_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Build sample records for the dataset card.

        Args:
            dataset_df: The dataset as a pandas DataFrame.

        Returns:
            List of sample records as dictionaries.
        """
        num_samples = min(5, len(dataset_df))
        if num_samples == 0:
            return []

        sample_df = dataset_df.head(num_samples)
        records = sample_df.to_dict(orient="records")
        return [
            {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for k, v in record.items()}
            for record in records
        ]

    def _build_config_types_summary(self, column_configs: list[Any] | None) -> dict[str, int]:
        """Build summary of configuration types.

        Args:
            column_configs: List of column configurations.

        Returns:
            Dictionary mapping config type names to counts.
        """
        if not column_configs:
            return {}

        config_types: dict[str, int] = {}
        for col_config in column_configs:
            config_type = type(col_config).__name__
            config_types[config_type] = config_types.get(config_type, 0) + 1
        return config_types

    def _build_column_stats_by_type(self, analysis: DatasetProfilerResults) -> dict[str, list[dict[str, Any]]]:
        """Build column statistics grouped by type.

        Args:
            analysis: Profiling analysis results.

        Returns:
            Dictionary mapping column types to lists of statistics dictionaries.
        """
        column_stats_by_type: dict[str, list[Any]] = {}
        for column_type in analysis.column_types:
            try:
                column_type_enum = DataDesignerColumnType(column_type)
                stats = analysis.get_column_statistics_by_type(column_type_enum)
                if stats:
                    column_stats_by_type[column_type] = stats
            except (ValueError, TypeError):
                continue

        return {
            col_type: [pydantic_to_dict(stat) for stat in stats_list]
            for col_type, stats_list in column_stats_by_type.items()
        }

    def _sort_column_types(self, column_stats_by_type: dict[str, list[dict[str, Any]]]) -> list[str]:
        """Sort column types by display order.

        Args:
            column_stats_by_type: Dictionary mapping column types to statistics.

        Returns:
            Sorted list of column type names.
        """
        display_order = get_column_display_order()
        return sorted(
            column_stats_by_type.keys(),
            key=lambda x: display_order.index(x) if x in display_order else len(display_order),
        )

    def _create_dataset_card(
        self,
        dataset_df: pd.DataFrame,
        template_variables: dict[str, Any],
    ) -> DataDesignerDatasetCard:
        """Create a dataset card from template variables.

        Args:
            dataset_df: The dataset as a pandas DataFrame.
            template_variables: Template variables for the card.

        Returns:
            DataDesignerDatasetCard instance.
        """
        tags_list = ["datadesigner", "synthetic"]
        return DataDesignerDatasetCard.from_template(
            card_data=DatasetCardData(
                size_categories=parse_size_category(len(dataset_df)),
                tags=tags_list,
            ),
            tags=tags_list,
            **template_variables,
        )

    def _save_and_upload_card(
        self,
        card: DataDesignerDatasetCard,
        repo_id: str,
        token: str | None,
    ) -> None:
        """Save dataset card to temporary file and upload to hub.

        Args:
            card: The dataset card to upload.
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.

        Raises:
            ArtifactStorageError: If card saving fails.
        """
        with TemporaryDirectory() as tmpdir:
            card_path = Path(tmpdir) / "README.md"
            try:
                card.save(filepath=str(card_path))
            except Exception as e:
                raise ArtifactStorageError(f"Failed to save dataset card: {e}") from e

            HfApi(token=token).upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )

    @staticmethod
    def pull_from_hub(
        repo_id: str,
        *,
        token: str | None = None,
        split: str | None = None,
        include_analysis: bool = True,
        include_processors: bool = True,
        include_configs: bool = True,
        **kwargs: Any,
    ) -> HubDatasetResults:
        """Load a dataset and all associated artifacts from Hugging Face Hub.

        This function loads a dataset from the Hugging Face Hub along with analysis results,
        processor datasets, processor artifacts, and configuration files if available.

        Args:
            repo_id: The ID of the Hugging Face Hub repository (e.g., "username/dataset-name").
            token: Hugging Face token for authentication. If None, will check environment
                variables HF_TOKEN or HUGGINGFACE_HUB_TOKEN.
            split: The split to load from the dataset. If None, the default split will be used.
            include_analysis: Whether to load analysis results. Defaults to True.
            include_processors: Whether to load processor datasets and artifacts. Defaults to True.
            include_configs: Whether to load configuration files. Defaults to True.
            **kwargs: Additional arguments to pass to `datasets.load_dataset()`.

        Returns:
            A HubDatasetResults object containing the dataset and all associated artifacts.
        """
        resolved_token = resolve_hf_token(token)
        hf_dataset = HuggingFaceHubClient._load_dataset_from_hub(repo_id, split, resolved_token, **kwargs)
        dataset_df = hf_dataset.to_pandas()

        analysis = None
        if include_analysis:
            analysis = HuggingFaceHubClient._load_analysis_from_hub(repo_id, resolved_token)

        processor_datasets = None
        processor_artifacts = None
        if include_processors:
            processor_datasets, processor_artifacts = HuggingFaceHubClient._load_processors_from_hub(
                repo_id, resolved_token
            )

        metadata = None
        column_configs = None
        model_configs = None
        if include_configs:
            metadata, column_configs, model_configs = HuggingFaceHubClient._load_configs_from_hub(
                repo_id, resolved_token
            )

        return HubDatasetResults(
            dataset=dataset_df,
            analysis=analysis,
            processor_datasets=processor_datasets,
            processor_artifacts=processor_artifacts,
            metadata=metadata,
            column_configs=column_configs,
            model_configs=model_configs,
        )

    @staticmethod
    def _load_dataset_from_hub(
        repo_id: str,
        split: str | None,
        token: str | None,
        **kwargs: Any,
    ) -> Dataset:
        """Load the main dataset from Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            split: The split to load. If None, the first split will be used.
            token: Hugging Face token for authentication.
            **kwargs: Additional arguments to pass to load_dataset.

        Returns:
            HuggingFace Dataset object.
        """
        hf_dataset = load_dataset(repo_id, split=split, token=token, **kwargs)

        if isinstance(hf_dataset, (DatasetDict, dict)):
            if split is None:
                split = next(iter(hf_dataset.keys()))
            hf_dataset = hf_dataset[split]

        return hf_dataset

    @staticmethod
    def _load_analysis_from_hub(
        repo_id: str,
        token: str | None,
    ) -> DatasetProfilerResults | None:
        """Load analysis results from Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.

        Returns:
            DatasetProfilerResults if available, None otherwise.
        """
        try:
            analysis_path = hf_hub_download(
                repo_id=repo_id,
                filename="analysis.json",
                repo_type="dataset",
                token=token,
            )
            with open(analysis_path, "r") as f:
                return DatasetProfilerResults.model_validate(json.load(f))
        except (HfHubHTTPError, FileNotFoundError, Exception):
            return None

    @staticmethod
    def _load_processors_from_hub(
        repo_id: str,
        token: str | None,
    ) -> tuple[dict[str, pd.DataFrame] | None, dict[str, Path] | None]:
        """Load processor datasets and artifacts from Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.

        Returns:
            Tuple of (processor_datasets dict, processor_artifacts dict), or (None, None) if unavailable.
        """
        try:
            repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
            processor_files = [f for f in repo_files if f.startswith("processors/")]

            processor_groups = HuggingFaceHubClient._group_processor_files(processor_files)
            processor_datasets = HuggingFaceHubClient._download_processor_datasets(repo_id, token, processor_groups)
            processor_artifacts = HuggingFaceHubClient._download_processor_artifacts(repo_id, token, processor_groups)

            return processor_datasets or None, processor_artifacts or None
        except (HfHubHTTPError, FileNotFoundError, Exception):
            return None, None

    @staticmethod
    def _group_processor_files(processor_files: list[str]) -> dict[str, list[str]]:
        """Group processor files by processor name.

        Args:
            processor_files: List of file paths in the processors/ directory.

        Returns:
            Dictionary mapping processor names to lists of file paths.
        """
        processor_groups: dict[str, list[str]] = {}
        for file_path in processor_files:
            parts = file_path.replace("processors/", "").split("/")
            processor_name = parts[0].replace(".parquet", "")
            if processor_name not in processor_groups:
                processor_groups[processor_name] = []
            processor_groups[processor_name].append(file_path)
        return processor_groups

    @staticmethod
    def _download_processor_datasets(
        repo_id: str,
        token: str | None,
        processor_groups: dict[str, list[str]],
    ) -> dict[str, pd.DataFrame]:
        """Download processor datasets from the hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.
            processor_groups: Dictionary mapping processor names to file paths.

        Returns:
            Dictionary mapping processor names to DataFrames.
        """
        processor_datasets: dict[str, pd.DataFrame] = {}
        for processor_name, files in processor_groups.items():
            parquet_file = next((f for f in files if f.endswith(".parquet")), None)
            if parquet_file:
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=parquet_file,
                        repo_type="dataset",
                        token=token,
                    )
                    processor_datasets[processor_name] = pd.read_parquet(local_path)
                except Exception:
                    pass
        return processor_datasets

    @staticmethod
    def _download_processor_artifacts(
        repo_id: str,
        token: str | None,
        processor_groups: dict[str, list[str]],
    ) -> dict[str, Path]:
        """Download processor artifacts from the hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.
            processor_groups: Dictionary mapping processor names to file paths.

        Returns:
            Dictionary mapping processor names to artifact directory paths.
        """
        processor_artifacts: dict[str, Path] = {}
        for processor_name, files in processor_groups.items():
            other_files = [f for f in files if not f.endswith(".parquet")]
            if other_files:
                with TemporaryDirectory() as tmpdir:
                    artifact_dir = Path(tmpdir) / processor_name
                    artifact_dir.mkdir(parents=True, exist_ok=True)

                    for artifact_file in other_files:
                        try:
                            local_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=artifact_file,
                                repo_type="dataset",
                                token=token,
                            )
                            relative_path = artifact_file.replace(f"processors/{processor_name}/", "")
                            if relative_path:
                                target_path = artifact_dir / relative_path
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(local_path, target_path)
                        except Exception:
                            pass

                    if any(artifact_dir.rglob("*")):
                        processor_artifacts[processor_name] = artifact_dir

        return processor_artifacts

    @staticmethod
    def _load_configs_from_hub(
        repo_id: str,
        token: str | None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        """Load configuration files from Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.

        Returns:
            Tuple of (metadata, column_configs, model_configs), with None values if unavailable.
        """
        metadata = HuggingFaceHubClient._load_metadata_from_hub(repo_id, token)
        column_configs = HuggingFaceHubClient._load_column_configs_from_hub(repo_id, token)
        model_configs = HuggingFaceHubClient._load_model_configs_from_hub(repo_id, token)

        return metadata, column_configs, model_configs

    @staticmethod
    def _load_metadata_from_hub(repo_id: str, token: str | None) -> dict[str, Any] | None:
        """Load metadata from Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.

        Returns:
            Metadata dictionary or None if unavailable.
        """
        try:
            metadata_path = hf_hub_download(
                repo_id=repo_id,
                filename="metadata.json",
                repo_type="dataset",
                token=token,
            )
            with open(metadata_path, "r") as f:
                return json.load(f)
        except (HfHubHTTPError, FileNotFoundError, Exception):
            return None

    @staticmethod
    def _load_column_configs_from_hub(repo_id: str, token: str | None) -> list[dict[str, Any]] | None:
        """Load column configurations from Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.

        Returns:
            List of column config dictionaries or None if unavailable.
        """
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="column_configs.json",
                repo_type="dataset",
                token=token,
            )
            with open(config_path, "r") as f:
                raw_column_configs = json.load(f)

            column_configs = []
            for config in raw_column_configs:
                if "columns" in config and isinstance(config["columns"], list):
                    column_configs.extend(config["columns"])
                else:
                    column_configs.append(config)
            return column_configs
        except (HfHubHTTPError, FileNotFoundError, Exception):
            return None

    @staticmethod
    def _load_model_configs_from_hub(repo_id: str, token: str | None) -> list[dict[str, Any]] | None:
        """Load model configurations from Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.

        Returns:
            List of model config dictionaries or None if unavailable.
        """
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_configs.json",
                repo_type="dataset",
                token=token,
            )
            with open(config_path, "r") as f:
                return json.load(f)
        except (HfHubHTTPError, FileNotFoundError, Exception):
            return None
