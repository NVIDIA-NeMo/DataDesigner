# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pandas as pd

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.default_model_settings import (
    get_default_model_configs,
    get_default_provider_name,
    get_default_providers,
    resolve_seed_default_model_settings,
)
from data_designer.config.interface import DataDesignerInterface
from data_designer.config.models import (
    ModelConfig,
    ModelProvider,
)
from data_designer.config.preview_results import PreviewResults
from data_designer.config.seed import LocalSeedDatasetReference
from data_designer.config.utils.constants import (
    DEFAULT_NUM_RECORDS,
    MODEL_CONFIGS_FILE_PATH,
    MODEL_PROVIDERS_FILE_PATH,
)
from data_designer.config.utils.info import InterfaceInfo
from data_designer.config.utils.io_helpers import write_seed_dataset
from data_designer.config.utils.misc import can_run_data_designer_locally
from data_designer.engine.analysis.dataset_profiler import (
    DataDesignerDatasetProfiler,
    DatasetProfilerConfig,
)
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.dataset_builders.column_wise_builder import ColumnWiseDatasetBuilder
from data_designer.engine.dataset_builders.utils.config_compiler import compile_dataset_builder_column_configs
from data_designer.engine.model_provider import resolve_model_provider_registry
from data_designer.engine.models.registry import create_model_registry
from data_designer.engine.resources.managed_storage import init_managed_blob_storage
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_dataset_data_store import (
    HfHubSeedDatasetDataStore,
    LocalSeedDatasetDataStore,
)
from data_designer.engine.secret_resolver import EnvironmentResolver, SecretResolver
from data_designer.interface.errors import (
    DataDesignerGenerationError,
    DataDesignerProfilingError,
    InvalidBufferValueError,
)
from data_designer.interface.results import DatasetCreationResults
from data_designer.logging import RandomEmoji

DEFAULT_BUFFER_SIZE = 1000

logger = logging.getLogger(__name__)


# Resolve default model settings on import to ensure they are available when the library is used.
if can_run_data_designer_locally():
    resolve_seed_default_model_settings()


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
        blob_storage_path: Path to the blob storage directory. Note this parameter
            is temporary and will be removed after we update person sampling for the library.
    """

    def __init__(
        self,
        artifact_path: Path | str | None = None,
        *,
        model_providers: list[ModelProvider] | None = None,
        secret_resolver: SecretResolver = EnvironmentResolver(),
        blob_storage_path: Path | str | None = None,
    ):
        self._secret_resolver = secret_resolver
        self._artifact_path = Path(artifact_path) if artifact_path is not None else Path.cwd() / "artifacts"
        self._buffer_size = DEFAULT_BUFFER_SIZE
        self._blob_storage = (
            init_managed_blob_storage()
            if blob_storage_path is None
            else init_managed_blob_storage(str(blob_storage_path))
        )
        self._model_providers = model_providers or self.get_default_model_providers()
        self._model_provider_registry = resolve_model_provider_registry(
            self._model_providers, get_default_provider_name()
        )

    @staticmethod
    def make_seed_reference_from_file(file_path: str | Path) -> LocalSeedDatasetReference:
        """Create a seed dataset reference from an existing file.

        Supported file extensions: .parquet (recommended), .csv, .json, .jsonl

        Args:
            file_path: Path to an existing dataset file.

        Returns:
            A LocalSeedDatasetReference pointing to the specified file.
        """
        return LocalSeedDatasetReference(dataset=str(file_path))

    @classmethod
    def make_seed_reference_from_dataframe(
        cls, dataframe: pd.DataFrame, file_path: str | Path
    ) -> LocalSeedDatasetReference:
        """Create a seed dataset reference from a pandas DataFrame.

        This method writes the DataFrame to disk and returns a reference that can
        be passed to the config builder's `with_seed_dataset` method. If the file
        already exists, it will be overwritten.

        Supported file extensions: .parquet (recommended), .csv, .json, .jsonl

        Args:
            dataframe: Pandas DataFrame to use as seed data.
            file_path: Path where to save dataset.

        Returns:
            A LocalSeedDatasetReference pointing to the written file.
        """
        write_seed_dataset(dataframe, Path(file_path))
        return cls.make_seed_reference_from_file(file_path)

    @property
    def info(self) -> InterfaceInfo:
        """Get information about the Data Designer interface.

        Returns:
            InterfaceInfo object with information about the Data Designer interface.
        """
        return InterfaceInfo(model_providers=self._model_providers)

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
                folder name in the artifact path directory.

        Returns:
            DatasetCreationResults object with methods for loading the generated dataset,
            analysis results, and displaying sample records for inspection.

        Raises:
            DataDesignerGenerationError: If an error occurs during dataset generation.
            DataDesignerProfilingError: If an error occurs during dataset profiling.
        """
        logger.info("🎨 Creating Data Designer dataset")

        resource_provider = self._create_resource_provider(dataset_name, config_builder)

        builder = self._create_dataset_builder(config_builder, resource_provider)

        try:
            builder.build(num_records=num_records, buffer_size=self._buffer_size)
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

        return DatasetCreationResults(
            artifact_storage=builder.artifact_storage,
            analysis=analysis,
            config_builder=config_builder,
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
        builder = self._create_dataset_builder(config_builder, resource_provider)

        try:
            dataset = builder.build_preview(num_records=num_records)
        except Exception as e:
            raise DataDesignerGenerationError(f"🛑 Error generating preview dataset: {e}")

        try:
            profiler = self._create_dataset_profiler(config_builder, resource_provider)
            analysis = profiler.profile_dataset(num_records, dataset)
        except Exception as e:
            raise DataDesignerProfilingError(f"🛑 Error profiling preview dataset: {e}")

        if len(dataset) > 0 and isinstance(analysis, DatasetProfilerResults) and len(analysis.column_statistics) > 0:
            logger.info(f"{RandomEmoji.success()} Preview complete!")

        return PreviewResults(
            dataset=dataset,
            analysis=analysis,
            config_builder=config_builder,
        )

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

    def set_buffer_size(self, buffer_size: int) -> None:
        """Set the buffer size for dataset generation.

        The buffer size controls how many records are processed in memory at once
        during dataset generation using the `create` method. The default value is
        set to the constant `DEFAULT_BUFFER_SIZE` defined in the data_designer module.

        Args:
            buffer_size: Number of records to process in each buffer.

        Raises:
            InvalidBufferValueError: If buffer size is less than or equal to 0.
        """
        if buffer_size <= 0:
            raise InvalidBufferValueError("Buffer size must be greater than 0.")
        self._buffer_size = buffer_size

    def _create_dataset_builder(
        self, config_builder: DataDesignerConfigBuilder, resource_provider: ResourceProvider
    ) -> ColumnWiseDatasetBuilder:
        return ColumnWiseDatasetBuilder(
            column_configs=compile_dataset_builder_column_configs(config_builder.build(raise_exceptions=True)),
            processor_configs=config_builder.get_processor_configs(),
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
        model_configs = config_builder.model_configs
        ArtifactStorage.mkdir_if_needed(self._artifact_path)
        return ResourceProvider(
            artifact_storage=ArtifactStorage(artifact_path=self._artifact_path, dataset_name=dataset_name),
            model_registry=create_model_registry(
                model_configs=model_configs,
                model_provider_registry=self._model_provider_registry,
                secret_resolver=self._secret_resolver,
            ),
            blob_storage=self._blob_storage,
            datastore=(
                LocalSeedDatasetDataStore()
                if (settings := config_builder.get_seed_datastore_settings()) is None
                else HfHubSeedDatasetDataStore(
                    endpoint=settings.endpoint,
                    token=settings.token,
                )
            ),
        )
