# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.default_model_settings import (
    get_default_model_configs,
    get_default_providers,
    get_providers_with_missing_api_keys,
    resolve_seed_default_model_settings,
)
from data_designer.config.interface import DataDesignerInterface
from data_designer.config.mcp import MCPProviderT
from data_designer.config.models import (
    ModelConfig,
    ModelProvider,
)
from data_designer.config.preview_results import PreviewResults
from data_designer.config.run_config import JinjaRenderingEngine, RunConfig
from data_designer.config.utils.constants import (
    DEFAULT_NUM_RECORDS,
    MANAGED_ASSETS_PATH,
    MODEL_CONFIGS_FILE_PATH,
    MODEL_PROVIDERS_FILE_PATH,
)
from data_designer.config.utils.info import InfoType, InterfaceInfo
from data_designer.engine.analysis.dataset_profiler import DataDesignerDatasetProfiler, DatasetProfilerConfig
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.mcp.io import list_tool_names
from data_designer.engine.model_provider import ModelProviderRegistry, resolve_model_provider_registry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.errors import RETRYABLE_MODEL_ERRORS
from data_designer.engine.readiness import run_readiness_check
from data_designer.engine.resources.person_reader import (
    PersonReader,
    create_person_reader,
)
from data_designer.engine.resources.resource_provider import ResourceProvider, create_resource_provider
from data_designer.engine.resources.seed_reader import (
    AgentRolloutSeedReader,
    DataFrameSeedReader,
    DirectorySeedReader,
    FileContentsSeedReader,
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
from data_designer.engine.storage.artifact_storage import ArtifactStorage, ResumeMode
from data_designer.interface.composite_workflow import CompositeWorkflow
from data_designer.interface.errors import (
    DataDesignerEarlyShutdownError,
    DataDesignerGenerationError,
    DataDesignerProfilingError,
)
from data_designer.interface.results import DatasetCreationResults
from data_designer.logging import LOG_INDENT, RandomEmoji, configure_logging, is_logging_configured
from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController

logger = logging.getLogger(__name__)

_CHECK_MODELS_RETRYABLE_ERRORS = RETRYABLE_MODEL_ERRORS + (TimeoutError,)


def _initialize_interface_runtime(*, auto_configure_logging: bool = True) -> None:
    """Run idempotent runtime initialization for the interface package."""
    if auto_configure_logging and not is_logging_configured():
        configure_logging()
    resolve_seed_default_model_settings()


DEFAULT_SECRET_RESOLVER = CompositeResolver([EnvironmentResolver(), PlaintextResolver()])

DEFAULT_SEED_READERS = [
    HuggingFaceSeedReader(),
    LocalFileSeedReader(),
    DataFrameSeedReader(),
    DirectorySeedReader(),
    FileContentsSeedReader(),
    AgentRolloutSeedReader(),
]
for plugin in PluginRegistry().get_plugins(PluginType.SEED_READER):
    DEFAULT_SEED_READERS.append(plugin.impl_cls())


class DataDesigner(DataDesignerInterface[DatasetCreationResults]):
    """Main interface for creating datasets with Data Designer.

    This class provides the primary interface for building synthetic datasets using
    Data Designer configurations. It manages model providers, artifact storage, and
    orchestrates the dataset creation and profiling processes.

    Args:
        artifact_path: Path where generated artifacts will be stored. If not
            provided, artifacts are stored in an `artifacts` directory under the
            current working directory.
        model_providers: Optional list of model providers for LLM generation. If None,
            uses default providers.
        secret_resolver: Resolver for handling secrets and credentials. If None,
            uses the default composite resolver, which checks environment variables
            and plaintext values.
        seed_readers: Optional list of seed readers. If None, uses default readers.
        managed_assets_path: Path to the managed assets directory. This is used to point
            to the location of managed datasets and other assets used during dataset generation.
            If not provided, will check for an environment variable called DATA_DESIGNER_MANAGED_ASSETS_PATH.
            If the environment variable is not set, will use the default managed assets directory, which
            is defined in `data_designer.config.utils.constants`.
        person_reader: Optional custom reader for person datasets.
            If provided, this reader will be used instead of the default local reader.
            This allows clients to customize how managed datasets are accessed (e.g.,
            using custom fsspec clients for S3 or other remote storage).
        mcp_providers: Optional list of MCP provider configurations to enable tool-calling for
            LLM generation columns. Supports both MCPProvider (remote SSE or Streamable HTTP) and
            LocalStdioMCPProvider (local subprocess).
        auto_configure_logging: Whether to apply Data Designer's default logging
            configuration during construction. Set to False when embedding Data
            Designer in an application that manages its own logging — Data Designer
            will then leave existing root handlers and log levels untouched.
            Automatic configuration is also skipped when
            `data_designer.logging.configure_logging()` was already called.
    """

    def __init__(
        self,
        artifact_path: Path | str | None = None,
        *,
        model_providers: list[ModelProvider] | None = None,
        secret_resolver: SecretResolver | None = None,
        seed_readers: list[SeedReader] | None = None,
        managed_assets_path: Path | str | None = None,
        person_reader: PersonReader | None = None,
        mcp_providers: list[MCPProviderT] | None = None,
        auto_configure_logging: bool = True,
    ):
        _initialize_interface_runtime(auto_configure_logging=auto_configure_logging)
        self._secret_resolver = secret_resolver or DEFAULT_SECRET_RESOLVER
        self._artifact_path = Path(artifact_path) if artifact_path is not None else Path.cwd() / "artifacts"
        self._run_config = RunConfig()
        self._request_admission: AdaptiveRequestAdmissionController = self._create_request_admission_controller()
        self._managed_assets_path = Path(managed_assets_path or MANAGED_ASSETS_PATH)
        self._person_reader = person_reader
        self._model_providers = self._resolve_model_providers(model_providers)
        self._mcp_providers = mcp_providers or []
        self._model_provider_registry = resolve_model_provider_registry(self._model_providers)
        self._seed_reader_registry = SeedReaderRegistry(readers=seed_readers or DEFAULT_SEED_READERS)

    @property
    def info(self) -> InterfaceInfo:
        """Get information about the Data Designer interface.

        Returns:
            InterfaceInfo object with information about the Data Designer interface.
        """
        return self._get_interface_info(self._model_providers)

    @property
    def artifact_path(self) -> Path:
        """Directory where Data Designer writes artifacts by default."""
        return self._artifact_path

    def list_mcp_tool_names(self, mcp_provider_name: str, *, timeout_sec: float = 10.0) -> list[str]:
        """Connect to a configured MCP provider and return the names of its available tools.

        Args:
            mcp_provider_name: The ``name`` field of an MCP provider passed to the constructor.
            timeout_sec: Timeout in seconds for the MCP handshake. Defaults to 10.

        Returns:
            A list of tool name strings exposed by the MCP server.

        Raises:
            ValueError: If no provider with the given name was configured.
        """
        for provider in self._mcp_providers:
            if provider.name == mcp_provider_name:
                return list_tool_names(provider, timeout_sec=timeout_sec)
        configured = [p.name for p in self._mcp_providers]
        raise ValueError(f"No MCP provider named {mcp_provider_name!r}. Configured providers: {configured}")

    def create(
        self,
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int = DEFAULT_NUM_RECORDS,
        dataset_name: str = "dataset",
        resume: ResumeMode = ResumeMode.NEVER,
        artifact_path: Path | str | None = None,
        on_batch_complete: Callable[[Path], None] | None = None,
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
            resume: Controls how interrupted runs are handled.

                - ``ResumeMode.NEVER`` (default): always start a fresh generation run.
                - ``ResumeMode.ALWAYS``: resume from the last completed batch (sync) or row group
                  (async). ``buffer_size`` must match the original run. ``num_records`` may be
                  equal to or greater than what was already generated (you can extend the dataset);
                  ``num_records`` less than actual records so far raises ``DatasetGenerationError``.
                  If no checkpoint exists yet (interrupted before the first batch finished), silently
                  restarts from the beginning. Raises if the stored config is incompatible.
                - ``ResumeMode.IF_POSSIBLE``: like ``ALWAYS`` when the current config fingerprint
                  matches the stored config; otherwise starts a fresh run without raising an error.

                In all resume modes, in-flight partial results from the interrupted run are
                discarded before generation continues.
            artifact_path: Optional artifact root for this create call. Defaults
                to the path configured on this DataDesigner instance.
            on_batch_complete: Optional callback called with the completed batch artifact path after
                each batch is written. Useful for incremental workflows such as uploading each batch
                to remote storage, updating an external run monitor, or triggering downstream processing
                before the full dataset has finished. The callback runs synchronously in the generation
                path, so it is recommended to keep it lightweight or delegate slow work to a queue, e.g.
                ``on_batch_complete=lambda path: queue_upload(path)``. Callback exceptions abort the run
                and are wrapped as ``DataDesignerGenerationError``.

        Returns:
            DatasetCreationResults object with methods for loading the generated dataset,
            analysis results, and displaying sample records for inspection.

        Raises:
            DataDesignerGenerationError: If an error occurs during dataset generation.
            DataDesignerProfilingError: If an error occurs during dataset profiling.
        """
        logger.info("🎨 Creating Data Designer dataset")
        self._log_jinja_rendering_engine_mode()

        artifact_path = Path(artifact_path) if artifact_path is not None else self._artifact_path
        resource_provider = self._create_resource_provider(
            dataset_name,
            config_builder,
            resume=resume,
            artifact_path=artifact_path,
        )

        # ``DeprecationWarning`` is re-raised before the generic wrapper so that
        # ``warnings.warn(..., DeprecationWarning)`` calls inside the engine
        # surface their original message under strict warning filters
        # (``pytest.warns``, ``-W error::DeprecationWarning``, etc.) instead of being
        # swallowed and re-wrapped as a generic ``DataDesignerGenerationError``.
        try:
            builder = self._create_dataset_builder(config_builder.build(), resource_provider)
        except DeprecationWarning:
            raise
        except Exception as e:
            raise DataDesignerGenerationError(f"🛑 Error generating dataset: {e}") from e

        try:
            builder.build(num_records=num_records, on_batch_complete=on_batch_complete, resume=resume)
        except DeprecationWarning:
            raise
        except Exception as e:
            raise DataDesignerGenerationError(f"🛑 Error generating dataset: {e}") from e

        task_traces = builder.task_traces

        try:
            dataset_for_profiler = builder.artifact_storage.load_dataset_with_dropped_columns()
        except Exception as e:
            # Distinguish "early shutdown produced zero records" from generic load failures
            # so callers can react programmatically (e.g. retry on a different alias) instead
            # of parsing a wrapped FileNotFoundError. The scheduler's structured signal lives
            # on the builder for the duration of the run. We also require the run to have
            # produced zero records: a partial-salvage run that fails to load for unrelated
            # reasons (corrupt parquet, dropped-columns mismatch, filesystem hiccup) should
            # surface the original cause, not a misleading "zero records" diagnosis.
            if builder.early_shutdown and builder.actual_num_records == 0:
                raise DataDesignerEarlyShutdownError(
                    "🛑 Generation produced zero records — early shutdown was triggered. "
                    "The non-retryable error rate exceeded the configured threshold; check the "
                    "warnings above (and any 'Provider showing degraded performance' logs) for "
                    "the contributing failures."
                ) from e
            # Surface the original task error when the run produced 0 records due to a
            # deterministic non-retryable failure (e.g. bad seed source). Without this,
            # the user sees a generic FileNotFoundError-on-parquet that obscures the cause.
            root_cause = builder.first_non_retryable_error
            if root_cause is not None and builder.actual_num_records == 0:
                raise DataDesignerGenerationError(f"🛑 {type(root_cause).__name__}: {root_cause}") from root_cause
            raise DataDesignerGenerationError(
                f"🛑 Failed to load generated dataset — all records may have been dropped "
                f"due to generation failures. Check the warnings above for details. Original error: {e}"
            ) from e

        # Defensive: the batch manager skips writing when the buffer is empty, so in
        # practice load_dataset_with_dropped_columns() would raise before returning a
        # zero-row DataFrame. This guard protects against future changes to that contract.
        if len(dataset_for_profiler) == 0:
            # Mirror the load-failure guard above: only raise the typed error when
            # the run actually produced zero records. A partial-salvage run that
            # somehow returns an empty DF for unrelated reasons should surface the
            # generic error.
            if builder.early_shutdown and builder.actual_num_records == 0:
                raise DataDesignerEarlyShutdownError(
                    "🛑 Dataset is empty — early shutdown was triggered before any records "
                    "could complete. Check the warnings above for the contributing failures."
                )
            root_cause = builder.first_non_retryable_error
            if root_cause is not None and builder.actual_num_records == 0:
                raise DataDesignerGenerationError(f"🛑 {type(root_cause).__name__}: {root_cause}") from root_cause
            raise DataDesignerGenerationError(
                "🛑 Dataset is empty — all records were dropped due to generation failures. "
                "Check the warnings above for details on which columns failed."
            )

        try:
            profiler_column_configs = config_builder.get_column_configs() or builder.data_designer_config.columns
            profiler = self._create_dataset_profiler(
                config_builder,
                resource_provider,
                column_configs=profiler_column_configs,
            )
            analysis = profiler.profile_dataset(num_records, dataset_for_profiler)
        except Exception as e:
            raise DataDesignerProfilingError(f"🛑 Error profiling dataset: {e}") from e

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
            task_traces=task_traces,
        )

    async def acreate(
        self,
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int = DEFAULT_NUM_RECORDS,
        dataset_name: str = "dataset",
        resume: ResumeMode = ResumeMode.NEVER,
        artifact_path: Path | str | None = None,
    ) -> DatasetCreationResults:
        """Async wrapper for creating a dataset without blocking the caller's event loop."""
        kwargs = {"num_records": num_records, "dataset_name": dataset_name, "resume": resume}
        if artifact_path is not None:
            kwargs["artifact_path"] = artifact_path
        return await asyncio.to_thread(self.create, config_builder, **kwargs)

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
            DataDesignerEarlyShutdownError: If preview terminated via the early-shutdown gate
                with zero records produced. Subclass of ``DataDesignerGenerationError``.
            DataDesignerProfilingError: If an error occurs during preview dataset profiling.
        """
        logger.info(f"{RandomEmoji.previewing()} Preview generation in progress")
        self._log_jinja_rendering_engine_mode()

        resource_provider = self._create_resource_provider("preview-dataset", config_builder)
        try:
            builder = self._create_dataset_builder(config_builder.build(), resource_provider)
            raw_dataset = builder.build_preview(num_records=num_records)
            processed_dataset = builder.process_preview(raw_dataset)
        except DeprecationWarning:
            # See comment in create() — strict warning filters convert engine-level
            # ``warnings.warn(..., DeprecationWarning)`` into exceptions that we let
            # propagate untouched.
            raise
        except Exception as e:
            raise DataDesignerGenerationError(f"🛑 Error generating preview dataset: {e}") from e

        if len(processed_dataset) == 0:
            # Mirror the create() path: distinguish "early shutdown produced zero
            # records" from generic empty-dataset failures so callers can react
            # programmatically.
            if builder.early_shutdown and builder.actual_num_records == 0:
                raise DataDesignerEarlyShutdownError(
                    "🛑 Preview is empty — early shutdown was triggered before any records "
                    "could complete. Check the warnings above for the contributing failures."
                )
            root_cause = builder.first_non_retryable_error
            if root_cause is not None and builder.actual_num_records == 0:
                raise DataDesignerGenerationError(f"🛑 {type(root_cause).__name__}: {root_cause}") from root_cause
            raise DataDesignerGenerationError(
                "🛑 Dataset is empty — all records were dropped due to generation or processing failures. "
                "Check the warnings above for details on which columns failed."
            )

        dropped_columns = raw_dataset.columns.difference(processed_dataset.columns)
        if len(dropped_columns) > 0:
            dataset_for_profiler = lazy.pd.concat([processed_dataset, raw_dataset[dropped_columns]], axis=1)
        else:
            dataset_for_profiler = processed_dataset

        try:
            profiler = self._create_dataset_profiler(config_builder, resource_provider)
            analysis = profiler.profile_dataset(num_records, dataset_for_profiler)
        except Exception as e:
            raise DataDesignerProfilingError(f"🛑 Error profiling preview dataset: {e}") from e

        processor_artifacts: dict[str, list[dict]] = {}
        for name in builder.artifact_storage.list_processor_names():
            processor_artifacts[name] = builder.artifact_storage.load_processor_dataset(name).to_dict(orient="records")

        if isinstance(analysis, DatasetProfilerResults) and len(analysis.column_statistics) > 0:
            logger.info(f"{RandomEmoji.success()} Preview complete!")

        # Create dataset metadata from the resource provider
        dataset_metadata = resource_provider.get_dataset_metadata()

        return PreviewResults(
            dataset=processed_dataset,
            analysis=analysis,
            processor_artifacts=processor_artifacts,
            config_builder=config_builder,
            dataset_metadata=dataset_metadata,
            task_traces=builder.task_traces or None,
        )

    def compose_workflow(self, *, name: str) -> CompositeWorkflow:
        """Create an experimental composite workflow.

        Workflow chaining is experimental and its API, metadata schema, and
        artifact layout may change in future releases.

        Args:
            name: Workflow name used for the artifact directory.

        Returns:
            A composite workflow that can run named stages in sequence.
        """
        return CompositeWorkflow(name=name, data_designer=self)

    def _log_jinja_rendering_engine_mode(self) -> None:
        engine = JinjaRenderingEngine(self._run_config.jinja_rendering_engine)
        icon = "🔒" if engine == JinjaRenderingEngine.SECURE else "🏠"
        logger.info(f"{LOG_INDENT}{icon} Jinja rendering engine: {engine.value}")

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

    def check_models(
        self,
        config_builder: DataDesignerConfigBuilder,
        *,
        max_attempts: int = 1,
        retry_backoff_seconds: float = 0,
    ) -> None:
        """Probe every model and MCP tool referenced by the configuration.

        Runs the same readiness checks performed at the start of ``preview`` and
        ``create``: a tiny generation against each referenced model alias, plus a
        connectivity probe to each referenced MCP tool. Models whose ``ModelConfig``
        has ``skip_health_check=True`` are skipped.

        This complements :meth:`validate`: ``validate`` answers "is my configuration
        well-formed?", ``check_models`` answers "are the providers it depends on
        actually responsive?". Together they cover internal and external readiness
        without needing to start a workload.

        Args:
            config_builder: The DataDesignerConfigBuilder whose column configs
                determine which model aliases and tool aliases are probed.
            max_attempts: Total readiness attempts for retryable failures.
            retry_backoff_seconds: Base delay between attempts. The delay is
                multiplied by the completed attempt number.

        Returns:
            None if every (non-skipped) probe succeeded.

        Raises:
            ModelAuthenticationError, ModelNotFoundError, ModelAPIConnectionError,
                and other typed errors from ``data_designer.engine.models.errors``
                for any failing model probe.
            DatasetGenerationError: If a tool alias is referenced but no
                ``MCPRegistry`` is configured.
            TimeoutError: If async health-check execution exceeds 180 seconds.
            ValueError: If ``max_attempts`` is less than one or
                ``retry_backoff_seconds`` is negative.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")

        columns = config_builder.build().columns
        for attempt in range(1, max_attempts + 1):
            try:
                resource_provider = self._create_resource_provider("check-models", config_builder)
                run_readiness_check(
                    columns,
                    resource_provider,
                )
                return
            except _CHECK_MODELS_RETRYABLE_ERRORS as exc:
                if attempt == max_attempts:
                    raise
                delay = attempt * retry_backoff_seconds
                logger.warning(
                    f"Retrying model readiness check after {type(exc).__name__}: {exc} "
                    f"(attempt {attempt + 1}/{max_attempts}, sleeping {delay:g}s)"
                )
                time.sleep(delay)

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

    @property
    def model_provider_registry(self) -> ModelProviderRegistry:
        """Get the resolved model provider registry.

        Returns:
            The ModelProviderRegistry containing the providers resolved at
            construction time, either the user-supplied ``model_providers``
            list or the providers loaded from the YAML.
        """
        return self._model_provider_registry

    @property
    def run_config(self) -> RunConfig:
        """Get the runtime configuration applied to dataset generation.

        Returns:
            The active RunConfig instance. Note that ``RunConfig`` normalizes
            some fields on construction (e.g., ``shutdown_error_rate`` becomes
            ``1.0`` when ``disable_early_shutdown=True``), so the returned
            object may not exactly equal the one originally passed to
            ``set_run_config``.
        """
        return self._run_config

    def set_run_config(self, run_config: RunConfig) -> None:
        """Set the runtime configuration for dataset generation.

        Args:
            run_config: A RunConfig instance containing runtime settings such as
                early shutdown behavior, batch sizing via `buffer_size`, async task lease
                capacity via `max_in_flight_tasks`, and non-inference worker concurrency via
                `non_inference_max_parallel_workers`.

        Notes:
            When `disable_early_shutdown=True`, DataDesigner will never terminate generation early
            due to error-rate thresholds. Errors are still tracked for reporting.
        """
        self._run_config = run_config
        self._request_admission = self._create_request_admission_controller()

    def get_models(self, model_aliases: list[str]) -> dict[str, ModelFacade]:
        """Get a dict of ModelFacade instances for custom column development.

        Use this to experiment with custom column generator functions outside of
        the full pipeline. The returned dict matches the `models` argument passed
        to 3-arg custom column functions.

        Args:
            model_aliases: List of model aliases to include in the dict.

        Returns:
            Dict mapping alias to ModelFacade instance.
        """
        config_builder = DataDesignerConfigBuilder()
        resource_provider = self._create_resource_provider(
            "dev",
            config_builder,
            client_concurrency_mode=ClientConcurrencyMode.SYNC,
        )
        return {alias: resource_provider.model_registry.get_model(model_alias=alias) for alias in model_aliases}

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
                    "\n\t\t\tSee https://docs.nvidia.com/nemo/datadesigner/concepts/models/model-providers for more information."
                )
                self._get_interface_info(model_providers).display(InfoType.MODEL_PROVIDERS)
            return model_providers
        return model_providers or []

    def _create_dataset_builder(
        self,
        data_designer_config: DataDesignerConfig,
        resource_provider: ResourceProvider,
    ) -> DatasetBuilder:
        return DatasetBuilder(
            data_designer_config=data_designer_config,
            resource_provider=resource_provider,
        )

    def _create_dataset_profiler(
        self,
        config_builder: DataDesignerConfigBuilder,
        resource_provider: ResourceProvider,
        *,
        column_configs: list[ColumnConfigT] | None = None,
    ) -> DataDesignerDatasetProfiler:
        return DataDesignerDatasetProfiler(
            config=DatasetProfilerConfig(
                column_configs=column_configs or config_builder.get_column_configs(),
                column_profiler_configs=config_builder.get_profilers(),
            ),
            resource_provider=resource_provider,
        )

    def _create_resource_provider(
        self,
        dataset_name: str,
        config_builder: DataDesignerConfigBuilder,
        *,
        resume: ResumeMode = ResumeMode.NEVER,
        artifact_path: Path | None = None,
        client_concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.ASYNC,
    ) -> ResourceProvider:
        artifact_path = artifact_path or self._artifact_path
        ArtifactStorage.mkdir_if_needed(artifact_path)

        seed_dataset_source = None
        if (seed_config := config_builder.get_seed_config()) is not None:
            seed_dataset_source = seed_config.source

        return create_resource_provider(
            artifact_storage=ArtifactStorage(artifact_path=artifact_path, dataset_name=dataset_name, resume=resume),
            model_configs=config_builder.model_configs,
            secret_resolver=self._secret_resolver,
            model_provider_registry=self._model_provider_registry,
            person_reader=self._person_reader or create_person_reader(str(self._managed_assets_path)),
            seed_dataset_source=seed_dataset_source,
            seed_reader_registry=self._seed_reader_registry,
            run_config=self._run_config,
            mcp_providers=self._mcp_providers,
            tool_configs=config_builder.tool_configs,
            client_concurrency_mode=client_concurrency_mode,
            request_admission=self._request_admission,
        )

    def _create_request_admission_controller(self) -> AdaptiveRequestAdmissionController:
        from data_designer.engine.models.factory import create_request_admission_controller

        return create_request_admission_controller(self._run_config)

    def _get_interface_info(self, model_providers: list[ModelProvider]) -> InterfaceInfo:
        return InterfaceInfo(model_providers=model_providers)
