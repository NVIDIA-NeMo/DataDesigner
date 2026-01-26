# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.default_model_settings import (
    get_default_model_configs,
    get_default_provider_name,
    get_default_providers,
    get_providers_with_missing_api_keys,
)
from data_designer.config.interface import DataDesignerInterface
from data_designer.config.models import (
    ModelConfig,
    ModelProvider,
)
from data_designer.config.preview_results import PreviewResults
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.constants import (
    DEFAULT_NUM_RECORDS,
    MANAGED_ASSETS_PATH,
    MODEL_CONFIGS_FILE_PATH,
    MODEL_PROVIDERS_FILE_PATH,
)
from data_designer.config.utils.info import InfoType, InterfaceInfo
from data_designer.engine.analysis.dataset_profiler import DataDesignerDatasetProfiler, DatasetProfilerConfig
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.dataset_builders.column_wise_builder import ColumnWiseDatasetBuilder
from data_designer.engine.model_provider import resolve_model_provider_registry
from data_designer.engine.resources.managed_storage import init_managed_blob_storage
from data_designer.engine.resources.resource_provider import ResourceProvider, create_resource_provider
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    HuggingFaceSeedReader,
    LocalFileSeedReader,
    SeedReader,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import (
    CompositeResolver,
    EnvironmentResolver,
    PlaintextResolver,
    SecretResolver,
)
from data_designer.interface.errors import (
    DataDesignerGenerationError,
    DataDesignerProfilingError,
)
from data_designer.interface.results import DatasetCreationResults
from data_designer.lazy_heavy_imports import pd
from data_designer.logging import RandomEmoji
from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_SECRET_RESOLVER = CompositeResolver([EnvironmentResolver(), PlaintextResolver()])

DEFAULT_SEED_READERS = [
    HuggingFaceSeedReader(),
    LocalFileSeedReader(),
    DataFrameSeedReader(),
]
for plugin in PluginRegistry().get_plugins(PluginType.SEED_READER):
    DEFAULT_SEED_READERS.append(plugin.impl_cls())


class DataDesigner(DataDesignerInterface[DatasetCreationResults]):
    """Main interface for creating datasets with Data Designer.

    This class provides the primary interface for building synthetic datasets using
    Data Designer configurations. It manages model providers, artifact storage, and
    orchestrates the dataset creation and profiling processes.

    Args:
        artifact_path: Path where generated artifacts will be stored.
        dataset_name: Name for the generated dataset. Defaults to "dataset".
            This will be used as the dataset folder name in the artifact path.
        model_providers: Optional list of model providers for LLM generation. If None,
            uses default providers.
        secret_resolver: Resolver for handling secrets and credentials. Defaults to
            EnvironmentResolver which reads secrets from environment variables.
        seed_readers: Optional list of seed readers. If None, uses default readers.
        managed_assets_path: Path to the managed assets directory. This is used to point
            to the location of managed datasets and other assets used during dataset generation.
            If not provided, will check for an environment variable called DATA_DESIGNER_MANAGED_ASSETS_PATH.
            If the environment variable is not set, will use the default managed assets directory, which
            is defined in `data_designer.config.utils.constants`.
    """

    def __init__(
        self,
        artifact_path: Path | str | None = None,
        *,
        model_providers: list[ModelProvider] | None = None,
        secret_resolver: SecretResolver | None = None,
        seed_readers: list[SeedReader] | None = None,
        managed_assets_path: Path | str | None = None,
    ):
        self._secret_resolver = secret_resolver or DEFAULT_SECRET_RESOLVER
        self._artifact_path = Path(artifact_path) if artifact_path is not None else Path.cwd() / "artifacts"
        self._run_config = RunConfig()
        self._managed_assets_path = Path(managed_assets_path or MANAGED_ASSETS_PATH)
        self._model_providers = self._resolve_model_providers(model_providers)
        self._model_provider_registry = resolve_model_provider_registry(
            self._model_providers, get_default_provider_name()
        )
        self._seed_reader_registry = SeedReaderRegistry(readers=seed_readers or DEFAULT_SEED_READERS)

    @property
    def info(self) -> InterfaceInfo:
        """Get information about the Data Designer interface.

        Returns:
            InterfaceInfo object with information about the Data Designer interface.
        """
        return self._get_interface_info(self._model_providers)

    def create(
        self,
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int = DEFAULT_NUM_RECORDS,
        dataset_name: str = "dataset",
    ) -> DatasetCreationResults:
        """Create dataset and save results to the local artifact storage.

        This method orchestrates the full dataset creation pipeline including building
        the dataset according to the configuration, profiling the generated data, and
        storing artifacts.

        Args:
            config_builder: The DataDesignerConfigBuilder containing the dataset
                configuration (columns, constraints, seed data, etc.).
            num_records: Number of records to generate.
            dataset_name: Name of the dataset. This name will be used as the dataset
                folder name in the artifact path directory. If a non-empty directory with the
                same name already exists, dataset will be saved to a new directory with
                a datetime stamp. For example, if the dataset name is "awesome_dataset" and a directory
                with the same name already exists, the dataset will be saved to a new directory
                with the name "awesome_dataset_2025-01-01_12-00-00".

        Returns:
            DatasetCreationResults object with methods for loading the generated dataset,
            analysis results, and displaying sample records for inspection.

        Raises:
            DataDesignerGenerationError: If an error occurs during dataset generation.
            DataDesignerProfilingError: If an error occurs during dataset profiling.
        """
        logger.info("🎨 Creating Data Designer dataset")

        resource_provider = self._create_resource_provider(dataset_name, config_builder)

        builder = self._create_dataset_builder(config_builder.build(), resource_provider)

        try:
            builder.build(num_records=num_records)
        except Exception as e:
            raise DataDesignerGenerationError(f"🛑 Error generating dataset: {e}")

        try:
            profiler = self._create_dataset_profiler(config_builder, resource_provider)
            analysis = profiler.profile_dataset(
                num_records,
                builder.artifact_storage.load_dataset_with_dropped_columns(),
            )
        except Exception as e:
            raise DataDesignerProfilingError(f"🛑 Error profiling dataset: {e}")

        dataset_metadata = resource_provider.get_dataset_metadata()

        # Update metadata with column statistics from analysis
        if analysis:
            builder.artifact_storage.update_metadata(
                {"column_statistics": [stat.model_dump(mode="json") for stat in analysis.column_statistics]}
            )

        return DatasetCreationResults(
            artifact_storage=builder.artifact_storage,
            analysis=analysis,
            config_builder=config_builder,
            dataset_metadata=dataset_metadata,
        )

    def preview(
        self, config_builder: DataDesignerConfigBuilder, *, num_records: int = DEFAULT_NUM_RECORDS
    ) -> PreviewResults:
        """Generate preview dataset for fast iteration on your Data Designer configuration.

        All preview results are stored in memory. Once you are satisfied with the preview,
        use the `create` method to generate data at a larger scale and save results to disk.

        Args:
            config_builder: The DataDesignerConfigBuilder containing the dataset
                configuration (columns, constraints, seed data, etc.).
            num_records: Number of records to generate.

        Returns:
            PreviewResults object with methods for inspecting the results.

        Raises:
            DataDesignerGenerationError: If an error occurs during preview dataset generation.
            DataDesignerProfilingError: If an error occurs during preview dataset profiling.
        """
        logger.info(f"{RandomEmoji.previewing()} Preview generation in progress")

        resource_provider = self._create_resource_provider("preview-dataset", config_builder)
        builder = self._create_dataset_builder(config_builder.build(), resource_provider)

        try:
            raw_dataset = builder.build_preview(num_records=num_records)
            processed_dataset = builder.process_preview(raw_dataset)
        except Exception as e:
            raise DataDesignerGenerationError(f"🛑 Error generating preview dataset: {e}")

        dropped_columns = raw_dataset.columns.difference(processed_dataset.columns)
        if len(dropped_columns) > 0:
            dataset_for_profiler = pd.concat([processed_dataset, raw_dataset[dropped_columns]], axis=1)
        else:
            dataset_for_profiler = processed_dataset

        try:
            profiler = self._create_dataset_profiler(config_builder, resource_provider)
            analysis = profiler.profile_dataset(num_records, dataset_for_profiler)
        except Exception as e:
            raise DataDesignerProfilingError(f"🛑 Error profiling preview dataset: {e}")

        if builder.artifact_storage.processors_outputs_path.exists():
            processor_artifacts = {
                processor_config.name: pd.read_parquet(
                    builder.artifact_storage.processors_outputs_path / f"{processor_config.name}.parquet",
                    dtype_backend="pyarrow",
                ).to_dict(orient="records")
                for processor_config in config_builder.get_processor_configs()
            }
        else:
            processor_artifacts = {}

        if (
            len(processed_dataset) > 0
            and isinstance(analysis, DatasetProfilerResults)
            and len(analysis.column_statistics) > 0
        ):
            logger.info(f"{RandomEmoji.success()} Preview complete!")

        # Create dataset metadata from the resource provider
        dataset_metadata = resource_provider.get_dataset_metadata()

        return PreviewResults(
            dataset=processed_dataset,
            analysis=analysis,
            processor_artifacts=processor_artifacts,
            config_builder=config_builder,
            dataset_metadata=dataset_metadata,
        )

    def validate(self, config_builder: DataDesignerConfigBuilder) -> None:
        """Validate the Data Designer configuration as defined by the DataDesignerConfigBuilder
        with the configured engine components (SecretResolver, SeedReaders, etc.).

        Args:
            config_builder: The DataDesignerConfigBuilder containing the dataset
                configuration (columns, constraints, seed data, etc.).

        Returns:
            None if the configuration is valid.

        Raises:
            InvalidConfigError: If the configuration is invalid.
        """
        resource_provider = self._create_resource_provider("validate-configuration", config_builder)
        compile_data_designer_config(config_builder.build(), resource_provider)

    def get_default_model_configs(self) -> list[ModelConfig]:
        """Get the default model configurations.

        Returns:
            List of default model configurations.
        """
        logger.info(f"♻️ Using default model configs from {str(MODEL_CONFIGS_FILE_PATH)!r}")
        return get_default_model_configs()

    def get_default_model_providers(self) -> list[ModelProvider]:
        """Get the default model providers.

        Returns:
            List of default model providers.
        """
        logger.info(f"♻️ Using default model providers from {str(MODEL_PROVIDERS_FILE_PATH)!r}")
        return get_default_providers()

    @property
    def secret_resolver(self) -> SecretResolver:
        """Get the secret resolver used by this DataDesigner instance.

        Returns:
            The SecretResolver instance handling credentials and secrets.
        """
        return self._secret_resolver

    def set_run_config(self, run_config: RunConfig) -> None:
        """Set the runtime configuration for dataset generation.

        Args:
            run_config: A RunConfig instance containing runtime settings such as
                early shutdown behavior, batch sizing via `buffer_size`, and non-inference worker
                concurrency via `non_inference_max_parallel_workers`.

        Notes:
            When `disable_early_shutdown=True`, DataDesigner will never terminate generation early
            due to error-rate thresholds. Errors are still tracked for reporting.
        """
        self._run_config = run_config

    def _resolve_model_providers(self, model_providers: list[ModelProvider] | None) -> list[ModelProvider]:
        if model_providers is None:
            model_providers = get_default_providers()
            # Check which providers have missing API keys (from YAML file or env vars)
            providers_with_missing_keys = get_providers_with_missing_api_keys(model_providers)

            if len(providers_with_missing_keys) == len(model_providers):
                # All providers have missing API keys
                logger.warning(
                    "🚨 You are trying to use a default model provider but your API keys are missing."
                    "\n\t\t\tSet the API key for the default providers you intend to use and re-initialize the Data Designer object."
                    "\n\t\t\tAlternatively, you can provide your own model providers during Data Designer object initialization."
                    "\n\t\t\tSee https://nvidia-nemo.github.io/DataDesigner/concepts/models/model-providers/ for more information."
                )
                self._get_interface_info(model_providers).display(InfoType.MODEL_PROVIDERS)
            return model_providers
        return model_providers or []

    def _create_dataset_builder(
        self,
        data_designer_config: DataDesignerConfig,
        resource_provider: ResourceProvider,
    ) -> ColumnWiseDatasetBuilder:
        return ColumnWiseDatasetBuilder(
            data_designer_config=data_designer_config,
            resource_provider=resource_provider,
        )

    def _create_dataset_profiler(
        self, config_builder: DataDesignerConfigBuilder, resource_provider: ResourceProvider
    ) -> DataDesignerDatasetProfiler:
        return DataDesignerDatasetProfiler(
            config=DatasetProfilerConfig(
                column_configs=config_builder.get_column_configs(),
                column_profiler_configs=config_builder.get_profilers(),
            ),
            resource_provider=resource_provider,
        )

    def _create_resource_provider(
        self, dataset_name: str, config_builder: DataDesignerConfigBuilder
    ) -> ResourceProvider:
        ArtifactStorage.mkdir_if_needed(self._artifact_path)

        seed_dataset_source = None
        if (seed_config := config_builder.get_seed_config()) is not None:
            seed_dataset_source = seed_config.source

        return create_resource_provider(
            artifact_storage=ArtifactStorage(artifact_path=self._artifact_path, dataset_name=dataset_name),
            model_configs=config_builder.model_configs,
            secret_resolver=self._secret_resolver,
            model_provider_registry=self._model_provider_registry,
            blob_storage=init_managed_blob_storage(str(self._managed_assets_path)),
            seed_dataset_source=seed_dataset_source,
            seed_reader_registry=self._seed_reader_registry,
            run_config=self._run_config,
        )

    def _get_interface_info(self, model_providers: list[ModelProvider]) -> InterfaceInfo:
        return InterfaceInfo(model_providers=model_providers)
