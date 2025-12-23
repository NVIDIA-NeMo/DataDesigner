# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.integrations.huggingface.client import resolve_hf_token
from data_designer.integrations.huggingface.hub_results import HubDatasetResults

logger = logging.getLogger(__name__)


def reconstruct_dataset_creation_results(
    hub_results: HubDatasetResults,
    repo_id: str,
    artifact_path: Path | str | None = None,
    token: str | None = None,
) -> tuple[ArtifactStorage, DataDesignerConfigBuilder]:
    """Reconstruct ArtifactStorage and DataDesignerConfigBuilder from hub results.

    This function downloads all artifacts from the Hugging Face Hub and reconstructs
    the necessary components for a DatasetCreationResults object.

    Args:
        hub_results: Results from pulling from the hub.
        repo_id: The ID of the Hugging Face Hub repository.
        artifact_path: Optional path to save downloaded artifacts. If None, a temporary
            directory will be used.
        token: Hugging Face token for authentication.

    Returns:
        Tuple of (ArtifactStorage, DataDesignerConfigBuilder).

    Raises:
        ArtifactStorageError: If analysis results are not found or reconstruction fails.
    """
    if hub_results.analysis is None:
        raise ArtifactStorageError("Cannot reconstruct DatasetCreationResults: analysis results not found in hub.")

    if artifact_path is None:
        temp_dir = tempfile.mkdtemp(prefix="data_designer_hub_")
        artifact_path = Path(temp_dir)
    else:
        artifact_path = Path(artifact_path)
        artifact_path.mkdir(parents=True, exist_ok=True)

    dataset_name = "dataset"
    artifact_storage = ArtifactStorage(
        artifact_path=artifact_path,
        dataset_name=dataset_name,
    )
    base_path = artifact_storage.base_dataset_path
    base_path.mkdir(parents=True, exist_ok=True)

    _save_main_dataset(hub_results, artifact_storage)
    _save_metadata(hub_results, base_path)
    _save_processor_datasets(hub_results, base_path)
    _save_processor_artifacts(hub_results, base_path)
    _save_config_files(hub_results, base_path)

    config_builder = _reconstruct_config_builder(hub_results, repo_id, token)

    return artifact_storage, config_builder


def _save_main_dataset(hub_results: HubDatasetResults, artifact_storage: ArtifactStorage) -> None:
    """Save the main dataset as parquet files.

    Args:
        hub_results: Results from pulling from the hub.
        artifact_storage: Artifact storage object.
    """
    final_dataset_path = artifact_storage.final_dataset_path
    final_dataset_path.mkdir(parents=True, exist_ok=True)
    hub_results.dataset.to_parquet(final_dataset_path / "data.parquet", index=False)


def _save_metadata(hub_results: HubDatasetResults, base_path: Path) -> None:
    """Save metadata if available.

    Args:
        hub_results: Results from pulling from the hub.
        base_path: Base path for artifacts.
    """
    if hub_results.metadata:
        with open(base_path / "metadata.json", "w") as f:
            json.dump(hub_results.metadata, f, indent=2)


def _save_processor_datasets(hub_results: HubDatasetResults, base_path: Path) -> None:
    """Save processor datasets if available.

    Args:
        hub_results: Results from pulling from the hub.
        base_path: Base path for artifacts.
    """
    if not hub_results.processor_datasets:
        return

    processors_path = base_path / "processors-files"
    processors_path.mkdir(parents=True, exist_ok=True)
    for processor_name, processor_df in hub_results.processor_datasets.items():
        processor_dir = processors_path / processor_name
        processor_dir.mkdir(parents=True, exist_ok=True)
        processor_df.to_parquet(processor_dir / f"{processor_name}.parquet", index=False)


def _save_processor_artifacts(hub_results: HubDatasetResults, base_path: Path) -> None:
    """Save processor artifacts if available.

    Args:
        hub_results: Results from pulling from the hub.
        base_path: Base path for artifacts.
    """
    if not hub_results.processor_artifacts:
        return

    processors_path = base_path / "processors-files"
    processors_path.mkdir(parents=True, exist_ok=True)
    for processor_name, artifact_dir in hub_results.processor_artifacts.items():
        if not artifact_dir.exists():
            continue
        target_dir = processors_path / processor_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(artifact_dir, target_dir)


def _save_config_files(hub_results: HubDatasetResults, base_path: Path) -> None:
    """Save configuration files if available.

    Args:
        hub_results: Results from pulling from the hub.
        base_path: Base path for artifacts.
    """
    if hub_results.column_configs:
        with open(base_path / "column_configs.json", "w") as f:
            json.dump(hub_results.column_configs, f, indent=2)

    if hub_results.model_configs:
        with open(base_path / "model_configs.json", "w") as f:
            json.dump(hub_results.model_configs, f, indent=2)


def _reconstruct_config_builder(
    hub_results: HubDatasetResults,
    repo_id: str,
    token: str | None,
) -> DataDesignerConfigBuilder:
    """Reconstruct the config builder from hub results or hub files.

    Args:
        hub_results: Results from pulling from the hub.
        repo_id: The ID of the Hugging Face Hub repository.
        token: Hugging Face token for authentication.

    Returns:
        DataDesignerConfigBuilder instance.
    """
    if hub_results.column_configs and hub_results.model_configs:
        model_configs = [ModelConfig.model_validate(mc) for mc in hub_results.model_configs]
        config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
        column_config_class_mapping = _get_column_config_class_mapping()

        for col_config_dict in hub_results.column_configs:
            configs_to_add = (
                col_config_dict["columns"]
                if "columns" in col_config_dict and isinstance(col_config_dict["columns"], list)
                else [col_config_dict]
            )
            for single_col_config_dict in configs_to_add:
                col_config = _load_column_config(single_col_config_dict, column_config_class_mapping)
                if col_config is not None:
                    config_builder.add_column(col_config)

        return config_builder

    resolved_token = resolve_hf_token(token)
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
        return DataDesignerConfigBuilder(model_configs=model_configs)
    except Exception:
        return DataDesignerConfigBuilder()


def _get_column_config_class_mapping() -> dict[str, type[Any]]:
    """Build a mapping from column_type string to config class dynamically.

    Returns:
        Dictionary mapping column type strings to config classes.
    """
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

    plugin_manager = PluginManager()
    for plugin in plugin_manager.get_column_generator_plugins():
        mapping[plugin.name] = plugin.config_cls

    return mapping


def _load_column_config(
    col_config_dict: dict[str, Any],
    column_config_class_mapping: dict[str, type[Any]],
) -> Any | None:
    """Load a single column config from dict using dynamic class mapping.

    Args:
        col_config_dict: Dictionary representation of column config.
        column_config_class_mapping: Mapping from column type to config class.

    Returns:
        Column config instance or None if loading fails.
    """
    column_type = col_config_dict.get("column_type")
    if not column_type:
        return None

    config_class = column_config_class_mapping.get(column_type)
    if config_class is None:
        logger.warning(
            f"Skipping column config with unknown type '{column_type}': {col_config_dict.get('name', 'unknown')}"
        )
        return None

    try:
        return config_class.model_validate(col_config_dict)
    except Exception as e:
        logger.warning(f"Failed to load column config '{col_config_dict.get('name', 'unknown')}': {e}. Skipping.")
        return None
