# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, get_token, hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.interface.huggingface.dataset_card import DataDesignerDatasetCard
from data_designer.interface.huggingface.hub_results import HubDatasetResults


def _resolve_hf_token(token: str | None) -> str | None:
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

    # Try to get token from huggingface_hub (checks env vars, cache, config file, etc.)
    try:
        token = get_token()
        if token:
            return token
    except Exception:
        # If get_token fails, continue to return None
        pass

    # Return None - huggingface_hub will handle authentication if user is logged in
    return None


def _size_categories_parser(num_records: int) -> str:
    """Parse dataset size into Hugging Face size category.

    Uses the same category names as Argilla's size_categories_parser.

    Args:
        num_records: Number of records in the dataset.

    Returns:
        Size category string matching Hugging Face format (e.g., "n<1K", "1K<n<10K", etc.).
    """
    AVAILABLE_SIZE_CATEGORIES = {
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

    for size, category in AVAILABLE_SIZE_CATEGORIES.items():
        if num_records < size:
            return category
    return "n>1T"


def _build_card_template_variables(
    dataset_df: pd.DataFrame,
    analysis: Any,
    config_builder: Any,
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

    # Prepare column information
    unconfigured_columns = {}
    all_columns = {}
    # Always populate all_columns for template use
    for col_name in sorted(column_names):
        all_columns[col_name] = str(dataset_df[col_name].dtype)

    if column_configs:
        configured_names = {col.name for col in column_configs}
        unconfigured = column_names - configured_names
        for col_name in sorted(unconfigured):
            unconfigured_columns[col_name] = str(dataset_df[col_name].dtype)

    # Prepare sample records
    num_samples = min(5, len(dataset_df))
    sample_records = []
    if num_samples > 0:
        sample_df = dataset_df.head(num_samples)
        records = sample_df.to_dict(orient="records")
        for record in records:
            # Convert to JSON-serializable format, handling complex types
            serializable_record = {}
            for k, v in record.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    serializable_record[k] = v
                else:
                    # Convert complex types to string representation
                    serializable_record[k] = str(v)
            sample_records.append(serializable_record)

    # Convert column_configs to dicts for safer template rendering
    column_configs_dicts = []
    if column_configs:
        for col_config in column_configs:
            if hasattr(col_config, "model_dump"):
                # Pydantic model
                config_dict = col_config.model_dump(mode="json")
                # Convert column_type enum to dict if it exists
                if "column_type" in config_dict and hasattr(config_dict["column_type"], "value"):
                    config_dict["column_type"] = {"value": config_dict["column_type"].value}
                elif "column_type" in config_dict and not isinstance(config_dict["column_type"], dict):
                    # If it's an enum object, convert it
                    col_type = getattr(col_config, "column_type", None)
                    if col_type and hasattr(col_type, "value"):
                        config_dict["column_type"] = {"value": col_type.value}
                    else:
                        config_dict["column_type"] = {"value": str(config_dict.get("column_type", "unknown"))}
                column_configs_dicts.append(config_dict)
            elif hasattr(col_config, "__dict__"):
                # Regular object - convert to dict
                config_dict = {}
                for key in dir(col_config):
                    if not key.startswith("_") and not callable(getattr(col_config, key, None)):
                        try:
                            value = getattr(col_config, key, None)
                            if isinstance(value, (str, int, float, bool, type(None))):
                                config_dict[key] = value
                            elif hasattr(value, "value"):  # Enum
                                config_dict[key] = {"value": value.value} if key == "column_type" else value.value
                            else:
                                config_dict[key] = str(value) if value is not None else None
                        except Exception:
                            pass
                column_configs_dicts.append(config_dict)
            else:
                column_configs_dicts.append(col_config)

    # Prepare config types summary
    config_types: dict[str, int] = {}
    if column_configs:
        for col_config in column_configs:
            config_type = type(col_config).__name__
            config_types[config_type] = config_types.get(config_type, 0) + 1

    # Group column statistics by type
    from data_designer.config.column_types import DataDesignerColumnType, get_column_display_order

    column_stats_by_type: dict[str, list] = {}
    display_order = get_column_display_order()

    for column_type in analysis.column_types:
        # column_type is already a string, convert to DataDesignerColumnType enum
        try:
            column_type_enum = DataDesignerColumnType(column_type)
        except (ValueError, TypeError):
            # Skip invalid column types
            continue
        stats = analysis.get_column_statistics_by_type(column_type_enum)
        if stats:
            # Convert stat objects to dicts for safer template rendering
            stats_dicts = []
            for stat in stats:
                if hasattr(stat, "model_dump"):
                    # Pydantic model - convert to dict
                    stat_dict = stat.model_dump(mode="json")
                    # Handle enum fields like sampler_type
                    if "sampler_type" in stat_dict and not isinstance(stat_dict["sampler_type"], (str, dict)):
                        sampler_type = getattr(stat, "sampler_type", None)
                        if sampler_type and hasattr(sampler_type, "value"):
                            stat_dict["sampler_type"] = {"value": sampler_type.value}
                        else:
                            stat_dict["sampler_type"] = {"value": str(stat_dict.get("sampler_type", "unknown"))}
                    stats_dicts.append(stat_dict)
                elif hasattr(stat, "__dict__"):
                    # Regular object - convert to dict
                    stat_dict = {}
                    for key in dir(stat):
                        if not key.startswith("_") and not callable(getattr(stat, key, None)):
                            try:
                                value = getattr(stat, key, None)
                                if isinstance(value, (str, int, float, bool, type(None))):
                                    stat_dict[key] = value
                                elif hasattr(value, "value"):  # Enum
                                    # For enums, store as dict with value key for consistency
                                    stat_dict[key] = (
                                        {"value": value.value}
                                        if key in ["sampler_type", "column_type"]
                                        else value.value
                                    )
                                else:
                                    stat_dict[key] = str(value) if value is not None else None
                            except Exception:
                                pass
                    stats_dicts.append(stat_dict)
                else:
                    stats_dicts.append(stat)
            column_stats_by_type[column_type] = stats_dicts

    # Sort column types by display order
    sorted_column_types = sorted(
        column_stats_by_type.keys(),
        key=lambda x: display_order.index(x) if x in display_order else len(display_order),
    )

    # Convert column_statistics to dicts for safer template rendering
    column_statistics_dicts = []
    if analysis.column_statistics:
        for stat in analysis.column_statistics:
            if hasattr(stat, "model_dump"):
                # Pydantic model
                column_statistics_dicts.append(stat.model_dump(mode="json"))
            elif hasattr(stat, "__dict__"):
                # Regular object - convert to dict
                stat_dict = {}
                for key in dir(stat):
                    if not key.startswith("_") and not callable(getattr(stat, key, None)):
                        try:
                            value = getattr(stat, key, None)
                            if isinstance(value, (str, int, float, bool, type(None))):
                                stat_dict[key] = value
                            elif hasattr(value, "value"):  # Enum
                                stat_dict[key] = value.value
                            else:
                                stat_dict[key] = str(value) if value is not None else None
                        except Exception:
                            pass
                column_statistics_dicts.append(stat_dict)
            else:
                column_statistics_dicts.append(stat)

    return {
        "size_categories": _size_categories_parser(len(dataset_df)),
        "num_records": len(dataset_df),
        "target_num_records": analysis.target_num_records,
        "percent_complete": analysis.percent_complete,
        "num_columns": len(dataset_df.columns),
        "repo_id": repo_id,
        "metadata": metadata or {},
        "column_configs": column_configs_dicts if column_configs_dicts else [],
        "unconfigured_columns": unconfigured_columns,
        "all_columns": all_columns,
        "column_statistics": column_statistics_dicts,
        "column_stats_by_type": column_stats_by_type,
        "sorted_column_types": sorted_column_types,
        "num_samples": num_samples,
        "sample_records": sample_records,
        "config_types": config_types,
    }


class HasDataset(Protocol):
    """Protocol for classes that have a load_dataset method."""

    def load_dataset(self) -> pd.DataFrame: ...


class HasArtifactStorage(Protocol):
    """Protocol for classes that have artifact_storage with metadata_file_path."""

    @property
    def artifact_storage(self) -> Any: ...


class HuggingFaceHubMixin:
    """Mixin class for pushing and pulling datasets to/from Hugging Face Hub.

    This mixin provides the `push_to_hub` and `pull_from_hub` methods to classes that implement
    the `HasDataset` and `HasArtifactStorage` protocols.
    """

    def push_to_hub(
        self: Any,
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
        # Resolve token
        resolved_token = self._resolve_token(token)

        # Load dataset
        dataset_df = self.load_dataset()

        # Convert pandas DataFrame to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(dataset_df)

        # Push dataset to hub
        hf_dataset.push_to_hub(repo_id, token=resolved_token, **kwargs)

        # Push additional artifacts (analysis, processor datasets, configs)
        # Pass the repo_id so we can list actual files for metadata sanitization
        self._upload_additional_artifacts(repo_id, resolved_token)

        # Generate and upload dataset card if requested
        if generate_card:
            self._upload_dataset_card(repo_id, resolved_token, dataset_df)

    def _resolve_token(self, token: str | None) -> str | None:
        """Resolve the Hugging Face token from parameter, environment variables, or huggingface_hub.

        Args:
            token: Token provided as parameter.

        Returns:
            Resolved token or None if not found.
        """
        return _resolve_hf_token(token)

    def _sanitize_metadata_file_paths(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize file paths in metadata by converting local paths to remote paths.

        Args:
            metadata: Metadata dictionary that may contain file_paths.

        Returns:
            Metadata dictionary with sanitized file paths.
        """
        if "file_paths" not in metadata or not isinstance(metadata["file_paths"], list):
            return metadata

        sanitized_paths = []
        base_path = self.artifact_storage.base_dataset_path

        for file_path in metadata["file_paths"]:
            path_str = str(file_path)
            path_obj = Path(path_str)

            # Try to get relative path from base_dataset_path
            try:
                if path_obj.is_absolute():
                    try:
                        relative_path = path_obj.relative_to(base_path)
                        remote_path = f"data/{relative_path.as_posix()}"
                        sanitized_paths.append(remote_path)
                        continue
                    except ValueError:
                        # Path is not relative to base_path, try fallback
                        pass
            except Exception:
                # If Path operations fail, try string-based extraction
                pass

            # Fallback: extract directory structure from path string
            if "parquet-files" in path_str:
                idx = path_str.find("parquet-files")
                if idx != -1:
                    remaining = path_str[idx + len("parquet-files") :].lstrip("/\\")
                    sanitized_paths.append(f"data/parquet-files/{remaining}")
                else:
                    sanitized_paths.append(f"data/parquet-files/{path_obj.name}")
            else:
                sanitized_paths.append(f"data/{path_obj.name}")

        if sanitized_paths:
            metadata = metadata.copy()
            metadata["file_paths"] = sanitized_paths
        else:
            # If no paths could be sanitized, remove file_paths
            metadata = metadata.copy()
            metadata.pop("file_paths", None)

        return metadata

    def _upload_additional_artifacts(
        self: Any,
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
        hf_api = HfApi(token=token)

        # Get analysis from the instance
        analysis = getattr(self, "_analysis", None)

        # Upload analysis results
        if analysis is not None:
            try:
                analysis_json = analysis.model_dump(mode="json")
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
                # Log but don't fail if analysis can't be uploaded
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to upload analysis results: {e}")

        # Upload processor datasets and artifacts
        if hasattr(self, "artifact_storage") and hasattr(self.artifact_storage, "processors_outputs_path"):
            processors_path = self.artifact_storage.processors_outputs_path
            if processors_path.exists():
                self._upload_processor_artifacts(hf_api, repo_id, processors_path)

        # Upload metadata if it exists (sanitize file paths first)
        if hasattr(self, "artifact_storage") and hasattr(self.artifact_storage, "metadata_file_path"):
            metadata_path = self.artifact_storage.metadata_file_path
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    # Sanitize metadata: convert local file paths to remote Hugging Face Hub paths
                    metadata = self._sanitize_metadata_file_paths(metadata)

                    # Write sanitized metadata to temp file and upload
                    with TemporaryDirectory() as tmpdir:
                        sanitized_metadata_path = Path(tmpdir) / "metadata.json"
                        with open(sanitized_metadata_path, "w") as f:
                            json.dump(metadata, f, indent=2, default=str)
                        hf_api.upload_file(
                            path_or_fileobj=str(sanitized_metadata_path),
                            path_in_repo="metadata.json",
                            repo_id=repo_id,
                            repo_type="dataset",
                        )
                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to upload metadata: {e}")

        # Upload configuration files if they exist
        if hasattr(self, "artifact_storage") and hasattr(self.artifact_storage, "base_dataset_path"):
            base_path = self.artifact_storage.base_dataset_path
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
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(f"Failed to upload {config_file}: {e}")

    def _upload_processor_artifacts(
        self: Any,
        hf_api: HfApi,
        repo_id: str,
        processors_path: Path,
    ) -> None:
        """Upload processor datasets and artifacts to Hugging Face Hub.

        Args:
            hf_api: Hugging Face API client.
            repo_id: The ID of the Hugging Face Hub repository.
            processors_path: Path to the processors outputs directory.
        """
        # Find all processor directories
        processor_dirs = [d for d in processors_path.iterdir() if d.is_dir()]

        for processor_dir in processor_dirs:
            processor_name = processor_dir.name

            # Check if it's a dataset (contains parquet files)
            parquet_files = list(processor_dir.glob("*.parquet"))
            if parquet_files:
                # Upload as a dataset (combine all parquet files)
                try:
                    # Load all parquet files and combine
                    dfs = [pd.read_parquet(f) for f in parquet_files]
                    combined_df = pd.concat(dfs, ignore_index=True)

                    # Upload as a separate dataset file
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
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to upload processor dataset {processor_name}: {e}")

            # Upload other files in the processor directory as artifacts
            other_files = [f for f in processor_dir.rglob("*") if f.is_file() and f.suffix != ".parquet"]
            for artifact_file in other_files:
                try:
                    # Preserve directory structure relative to processor_dir
                    relative_path = artifact_file.relative_to(processors_path)
                    hf_api.upload_file(
                        path_or_fileobj=str(artifact_file),
                        path_in_repo=f"processors/{relative_path.as_posix()}",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to upload processor artifact {artifact_file}: {e}")

    def _upload_dataset_card(
        self: Any,
        repo_id: str,
        token: str | None,
        dataset_df: pd.DataFrame,
    ) -> None:
        """Generate and upload the dataset card to Hugging Face Hub.

        Args:
            repo_id: The ID of the Hugging Face Hub repository.
            token: Hugging Face token for authentication.
            dataset_df: The dataset as a pandas DataFrame.
        """
        # Get analysis and config_builder from the instance
        analysis = getattr(self, "_analysis", None)
        config_builder = getattr(self, "_config_builder", None)

        if analysis is None or config_builder is None:
            raise ArtifactStorageError(
                "Cannot generate dataset card: missing analysis or config_builder. "
                "Ensure the class has _analysis and _config_builder attributes."
            )

        # Load metadata if available and sanitize file paths
        metadata: dict[str, Any] | None = None
        if hasattr(self, "artifact_storage") and hasattr(self.artifact_storage, "metadata_file_path"):
            metadata_path = self.artifact_storage.metadata_file_path
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    # Sanitize file paths for the dataset card
                    metadata = self._sanitize_metadata_file_paths(metadata)
                except Exception:
                    # If metadata can't be loaded, continue without it
                    pass

        # Generate dataset card using from_template pattern (similar to Argilla)
        from huggingface_hub import DatasetCardData

        # Build template variables for the card
        template_variables = _build_card_template_variables(
            dataset_df=dataset_df,
            analysis=analysis,
            config_builder=config_builder,
            metadata=metadata,
            repo_id=repo_id,
        )
        # Ensure all_columns is always defined
        if "all_columns" not in template_variables:
            template_variables["all_columns"] = {}

        # Create card using DatasetCard.from_template with card_data and template_variables
        # DataDesignerDatasetCard extends DatasetCard and uses default_template_path
        # Unpack template_variables as kwargs for the template
        tags_list = ["datadesigner", "synthetic"]
        card = DataDesignerDatasetCard.from_template(
            card_data=DatasetCardData(
                size_categories=_size_categories_parser(len(dataset_df)),
                tags=tags_list,
            ),
            tags=tags_list,  # Also pass as template variable for explicit rendering
            **template_variables,
        )

        # Save card to temporary directory and upload
        with TemporaryDirectory() as tmpdir:
            card_path = Path(tmpdir) / "README.md"
            try:
                card.save(filepath=str(card_path))
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error saving dataset card: {e}")
                logger.error(f"Template variables keys: {list(template_variables.keys())}")
                # Try to identify which variable is causing the issue
                for key, value in template_variables.items():
                    if value is None:
                        logger.warning(f"Template variable '{key}' is None")
                raise
            hf_api = HfApi(token=token)
            hf_api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )

    @classmethod
    def pull_from_hub(
        cls: type[Any],
        repo_id: str,
        *,
        token: str | None = None,
        artifact_path: Path | str | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load a dataset and all artifacts from Hugging Face Hub as a DatasetCreationResults object.

        This classmethod downloads all artifacts from the Hugging Face Hub and reconstructs
        a DatasetCreationResults object that can be used just like one created from a local
        dataset generation run.

        Args:
            repo_id: The ID of the Hugging Face Hub repository (e.g., "username/dataset-name").
            token: Hugging Face token for authentication. If None, will check environment
                variables HF_TOKEN or HUGGINGFACE_HUB_TOKEN.
            artifact_path: Optional path to save downloaded artifacts. If None, a temporary
                directory will be used (note: temporary directories are cleaned up when
                the object is garbage collected).
            split: The split to load from the dataset. If None, the default split will be used.
            **kwargs: Additional arguments to pass to `pull_from_hub()` function.

        Returns:
            A DatasetCreationResults object containing the dataset, analysis, and all artifacts.

        Example:
            ```python
            from data_designer.interface.results import DatasetCreationResults

            # Load from hub (uses temporary directory)
            results = DatasetCreationResults.pull_from_hub("username/dataset-name")

            # Load to a specific directory
            results = DatasetCreationResults.pull_from_hub(
                "username/dataset-name",
                artifact_path="./downloaded_datasets/my_dataset"
            )

            # Access the dataset and analysis
            df = results.load_dataset()
            analysis = results.load_analysis()
            ```
        """
        import tempfile

        from data_designer.config.config_builder import DataDesignerConfigBuilder
        from data_designer.config.models import ModelConfig
        from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage

        # Pull all artifacts from hub using the function
        hub_results = pull_from_hub(
            repo_id=repo_id,
            token=token,
            split=split,
            include_analysis=True,
            include_processors=True,
            include_configs=True,
            **kwargs,
        )

        # Determine artifact path
        if artifact_path is None:
            # Use a temporary directory
            # Note: The directory will persist as long as the DatasetCreationResults object exists
            # Users should provide artifact_path for persistent storage
            temp_dir = tempfile.mkdtemp(prefix="data_designer_hub_")
            artifact_path = Path(temp_dir)
        else:
            artifact_path = Path(artifact_path)
            artifact_path.mkdir(parents=True, exist_ok=True)

        # Create artifact storage first to get the resolved dataset name
        dataset_name = "dataset"
        artifact_storage = ArtifactStorage(
            artifact_path=artifact_path,
            dataset_name=dataset_name,
        )
        base_path = artifact_storage.base_dataset_path
        base_path.mkdir(parents=True, exist_ok=True)

        # Save main dataset as parquet files
        final_dataset_path = artifact_storage.final_dataset_path
        final_dataset_path.mkdir(parents=True, exist_ok=True)
        hub_results.dataset.to_parquet(final_dataset_path / "data.parquet", index=False)

        # Save metadata if available
        if hub_results.metadata:
            metadata_path = base_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(hub_results.metadata, f, indent=2)

        # Save processor datasets and artifacts
        if hub_results.processor_datasets:
            processors_path = base_path / "processors-files"
            processors_path.mkdir(parents=True, exist_ok=True)
            for processor_name, processor_df in hub_results.processor_datasets.items():
                processor_dir = processors_path / processor_name
                processor_dir.mkdir(parents=True, exist_ok=True)
                processor_df.to_parquet(processor_dir / f"{processor_name}.parquet", index=False)

        # Copy processor artifacts if available
        if hub_results.processor_artifacts:
            processors_path = base_path / "processors-files"
            processors_path.mkdir(parents=True, exist_ok=True)
            import shutil

            for processor_name, artifact_dir in hub_results.processor_artifacts.items():
                if artifact_dir.exists():
                    target_dir = processors_path / processor_name
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(artifact_dir, target_dir)

        # Save config files
        if hub_results.column_configs:
            config_path = base_path / "column_configs.json"
            with open(config_path, "w") as f:
                json.dump(hub_results.column_configs, f, indent=2)

        if hub_results.model_configs:
            config_path = base_path / "model_configs.json"
            with open(config_path, "w") as f:
                json.dump(hub_results.model_configs, f, indent=2)

        # Reconstruct config builder from config files
        config_builder: DataDesignerConfigBuilder | None = None
        if hub_results.column_configs and hub_results.model_configs:
            # Load model configs
            model_configs = [ModelConfig.model_validate(mc) for mc in hub_results.model_configs]
            config_builder = DataDesignerConfigBuilder(model_configs=model_configs)

            # Build dynamic mapping from column_type to config class (includes plugins)
            def _get_column_config_class_mapping() -> dict[str, type[Any]]:
                """Build a mapping from column_type string to config class dynamically."""
                from data_designer.config.column_configs import (
                    ExpressionColumnConfig,
                    LLMCodeColumnConfig,
                    LLMJudgeColumnConfig,
                    LLMStructuredColumnConfig,
                    LLMTextColumnConfig,
                    SamplerColumnConfig,
                    SeedDatasetColumnConfig,
                    ValidationColumnConfig,
                )
                from data_designer.plugin_manager import PluginManager

                mapping: dict[str, type[Any]] = {
                    "sampler": SamplerColumnConfig,
                    "llm_text": LLMTextColumnConfig,
                    "llm_structured": LLMStructuredColumnConfig,
                    "llm_code": LLMCodeColumnConfig,
                    "llm_judge": LLMJudgeColumnConfig,
                    "expression": ExpressionColumnConfig,
                    "seed_dataset": SeedDatasetColumnConfig,
                    "validation": ValidationColumnConfig,
                }

                # Add plugin column configs dynamically
                plugin_manager = PluginManager()
                for plugin in plugin_manager.get_column_generator_plugins():
                    mapping[plugin.name] = plugin.config_cls

                return mapping

            column_config_class_mapping = _get_column_config_class_mapping()

            def _load_column_config(col_config_dict: dict[str, Any]) -> Any | None:
                """Load a single column config from dict using dynamic class mapping."""
                column_type = col_config_dict.get("column_type")
                if not column_type:
                    return None

                config_class = column_config_class_mapping.get(column_type)
                if config_class is None:
                    # Unknown column type - might be from a plugin or future version
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Skipping column config with unknown type '{column_type}': {col_config_dict.get('name', 'unknown')}"
                    )
                    return None

                try:
                    return config_class.model_validate(col_config_dict)
                except Exception as e:
                    # Skip columns that fail validation
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to load column config '{col_config_dict.get('name', 'unknown')}': {e}. Skipping."
                    )
                    return None

            for col_config_dict in hub_results.column_configs:
                # Handle MultiColumnConfig (has 'columns' key) by flattening it
                if "columns" in col_config_dict and isinstance(col_config_dict["columns"], list):
                    # This is a MultiColumnConfig - extract individual column configs
                    for single_col_config_dict in col_config_dict["columns"]:
                        col_config = _load_column_config(single_col_config_dict)
                        if col_config is not None:
                            config_builder.add_column(col_config)
                else:
                    # This is a single column config
                    single_col_config = _load_column_config(col_config_dict)
                    if single_col_config is not None:
                        config_builder.add_column(single_col_config)

        # If config builder couldn't be reconstructed, create a minimal one
        if config_builder is None:
            # Try to get model configs from environment or use defaults
            resolved_token = _resolve_hf_token(token)
            try:
                model_configs_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="model_configs.json",
                    repo_type="dataset",
                    token=resolved_token,
                )
                with open(model_configs_path, "r") as f:
                    model_configs_data = json.load(f)
                model_configs = [ModelConfig.model_validate(mc) for mc in model_configs_data]
                config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
            except Exception:
                # Fallback to default model configs
                config_builder = DataDesignerConfigBuilder()

        # Ensure we have analysis
        if hub_results.analysis is None:
            raise ArtifactStorageError("Cannot reconstruct DatasetCreationResults: analysis results not found in hub.")

        return cls(
            artifact_storage=artifact_storage,
            analysis=hub_results.analysis,
            config_builder=config_builder,
        )


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
    It is similar to Argilla's `from_hub` method but returns a comprehensive results object.

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

    Example:
        ```python
        from data_designer.interface.huggingface import pull_from_hub

        # Load a dataset with all artifacts from Hugging Face Hub
        results = pull_from_hub("username/dataset-name")
        df = results.dataset
        analysis = results.analysis
        processor_data = results.processor_datasets

        # Load only the main dataset
        results = pull_from_hub("username/dataset-name", include_analysis=False, include_processors=False)

        # Load a specific split
        results = pull_from_hub("username/dataset-name", split="train")
        ```
    """
    from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults

    # Resolve token
    resolved_token = _resolve_hf_token(token)

    # Load main dataset from hub
    hf_dataset = load_dataset(repo_id, split=split, token=resolved_token, **kwargs)

    # Handle DatasetDict
    if isinstance(hf_dataset, DatasetDict):
        if split is None:
            # Use the first split if no split specified
            split = next(iter(hf_dataset.keys()))
        hf_dataset = hf_dataset[split]
    elif isinstance(hf_dataset, dict):
        # Fallback for dict-like objects
        if split is None:
            split = next(iter(hf_dataset.keys()))
        hf_dataset = hf_dataset[split]

    # Convert to pandas DataFrame
    dataset_df = hf_dataset.to_pandas()

    # Load analysis results if requested
    analysis: DatasetProfilerResults | None = None
    if include_analysis:
        try:
            analysis_path = hf_hub_download(
                repo_id=repo_id,
                filename="analysis.json",
                repo_type="dataset",
                token=resolved_token,
            )
            with open(analysis_path, "r") as f:
                analysis_data = json.load(f)
            analysis = DatasetProfilerResults.model_validate(analysis_data)
        except (HfHubHTTPError, FileNotFoundError):
            # Analysis file may not exist, continue without it
            pass
        except Exception:
            # Other errors loading analysis, continue without it
            pass

    # Load processor datasets and artifacts if requested
    processor_datasets: dict[str, pd.DataFrame] | None = None
    processor_artifacts: dict[str, Path] | None = None
    if include_processors:
        try:
            repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=resolved_token)
            processor_files = [f for f in repo_files if f.startswith("processors/")]

            processor_datasets = {}
            processor_artifacts = {}

            # Group files by processor name
            processor_groups: dict[str, list[str]] = {}
            for file_path in processor_files:
                # Extract processor name from path like "processors/processor_name.parquet"
                # or "processors/processor_name/file.txt"
                parts = file_path.replace("processors/", "").split("/")
                processor_name = parts[0].replace(".parquet", "")

                if processor_name not in processor_groups:
                    processor_groups[processor_name] = []
                processor_groups[processor_name].append(file_path)

            # Download and load processor datasets
            for processor_name, files in processor_groups.items():
                parquet_files = [f for f in files if f.endswith(".parquet")]
                if parquet_files:
                    # Download the parquet file
                    parquet_file = parquet_files[0]  # Use first parquet file
                    try:
                        local_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=parquet_file,
                            repo_type="dataset",
                            token=resolved_token,
                        )
                        processor_datasets[processor_name] = pd.read_parquet(local_path)
                    except Exception:
                        pass

                # Download other artifacts
                other_files = [f for f in files if not f.endswith(".parquet")]
                if other_files:
                    # Download to a temporary directory
                    import shutil

                    with TemporaryDirectory() as tmpdir:
                        artifact_dir = Path(tmpdir) / processor_name
                        artifact_dir.mkdir(parents=True, exist_ok=True)

                        for artifact_file in other_files:
                            try:
                                local_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=artifact_file,
                                    repo_type="dataset",
                                    token=resolved_token,
                                )
                                # Preserve relative path structure
                                relative_path = artifact_file.replace(f"processors/{processor_name}/", "")
                                if relative_path:
                                    target_path = artifact_dir / relative_path
                                    target_path.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(local_path, target_path)
                            except Exception:
                                pass

                        # Only add if directory has files
                        if any(artifact_dir.rglob("*")):
                            # Copy to a persistent location or return the temp directory
                            # For now, we'll return the temp directory path
                            # Note: This will be cleaned up when tmpdir is deleted
                            # In a real implementation, you might want to copy to a user-specified location
                            processor_artifacts[processor_name] = artifact_dir

            if not processor_datasets:
                processor_datasets = None
            if not processor_artifacts:
                processor_artifacts = None
        except (HfHubHTTPError, FileNotFoundError):
            # Processors may not exist, continue without them
            pass
        except Exception:
            # Other errors loading processors, continue without them
            pass

    # Load configuration files if requested
    metadata: dict[str, Any] | None = None
    column_configs: list[dict[str, Any]] | None = None
    model_configs: list[dict[str, Any]] | None = None

    if include_configs:
        # Load metadata
        try:
            metadata_path = hf_hub_download(
                repo_id=repo_id,
                filename="metadata.json",
                repo_type="dataset",
                token=resolved_token,
            )
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (HfHubHTTPError, FileNotFoundError):
            pass
        except Exception:
            pass

        # Load column configs
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="column_configs.json",
                repo_type="dataset",
                token=resolved_token,
            )
            with open(config_path, "r") as f:
                raw_column_configs = json.load(f)
            # Flatten MultiColumnConfig objects (those with 'columns' key) into individual column configs
            column_configs = []
            for config in raw_column_configs:
                if "columns" in config and isinstance(config["columns"], list):
                    # This is a MultiColumnConfig - extract individual column configs
                    column_configs.extend(config["columns"])
                else:
                    # This is a single column config
                    column_configs.append(config)
        except (HfHubHTTPError, FileNotFoundError):
            pass
        except Exception:
            pass

        # Load model configs
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_configs.json",
                repo_type="dataset",
                token=resolved_token,
            )
            with open(config_path, "r") as f:
                model_configs = json.load(f)
        except (HfHubHTTPError, FileNotFoundError):
            pass
        except Exception:
            pass

    return HubDatasetResults(
        dataset=dataset_df,
        analysis=analysis,
        processor_datasets=processor_datasets,
        processor_artifacts=processor_artifacts,
        metadata=metadata,
        column_configs=column_configs,
        model_configs=model_configs,
    )

