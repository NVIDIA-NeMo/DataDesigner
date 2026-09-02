# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

from pydantic import ValidationError

import data_designer.lazy_heavy_imports as lazy
from data_designer.config import (
    DataDesignerConfigBuilder,
    LocalStdioMCPProvider,
    MCPProvider,
    ModelProvider,
    PartitionBlock,
    ResumeMode,
    RunConfig,
)
from data_designer.interface import DataDesigner
from data_designer.slurm.client.environment import PreparedClientEnvironment
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.filesystem import (
    compute_file_sha256,
    ensure_private_directory,
    publish_private_text,
    read_regular_bytes,
    replace_private_text,
)
from data_designer.slurm.client.records import (
    ClientEnvironmentManifest,
    ClientEnvironmentOutcome,
    ClientErrorCode,
    ClientOutcome,
    ClientPluginEntryPoint,
    ClientProgress,
    ClientProgressPhase,
    ClientResult,
)
from data_designer.slurm.config.environment import LiteralEnvironmentBinding, SecretRef
from data_designer.slurm.config.images import InstalledDistribution
from data_designer.slurm.config.run import LocalStdioMCPProviderConfig, RemoteMCPProviderConfig
from data_designer.slurm.contracts import ArtifactReference, compute_canonical_json_sha256
from data_designer.slurm.planning import PlannedShard, ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.state import CandidateOutcome, CandidateOutputFile, CandidateOutputManifest

Clock = Callable[[], datetime]
DataDesignerFactory = Callable[..., DataDesigner]
logger = logging.getLogger(__name__)


class CreationResults(Protocol):
    dataset_path: Path
    requested_num_records: int
    actual_num_records: int
    early_shutdown: bool | None
    requested_resume_mode: ResumeMode | None
    effective_resume_mode: ResumeMode | None

    def export(self, path: Path, *, format: str) -> Path:
        """Export the generated dataset."""


@dataclass(frozen=True)
class _ExecutionContext:
    plan: ResolvedSlurmRunPlan
    shard: PlannedShard
    builder: DataDesignerConfigBuilder
    designer: DataDesigner
    requested_resume: ResumeMode
    dataset_path: Path


@dataclass(frozen=True)
class _ValidatedCreation:
    requested_records: int
    actual_records: int
    dataset_path: Path
    expected_path: Path
    effective_resume: ResumeMode
    early_shutdown: bool


class _ProgressWriter:
    def __init__(
        self,
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
        clock: Clock,
        *,
        revision: int = 0,
    ) -> None:
        self._context = context
        self._prepared = prepared
        self._clock = clock
        self._revision = revision

    def required(
        self,
        phase: ClientProgressPhase,
        *,
        completed_records: int | None = None,
        error_code: ClientErrorCode | None = None,
    ) -> None:
        self._revision += 1
        progress = ClientProgress(
            schema_version=1,
            run_id=self._context.plan.run_id,
            shard_id=self._context.shard.shard_id,
            attempt_id=self._prepared.attempt_id,
            revision=self._revision,
            updated_at=self._clock(),
            phase=phase,
            requested_records=self._context.shard.requested_records,
            completed_records=completed_records,
            error_code=error_code,
        )
        replace_private_text(self._prepared.attempt_dir / "client-progress.json", progress.serialize_json())

    def best_effort(
        self,
        phase: ClientProgressPhase,
        *,
        completed_records: int | None = None,
        error_code: ClientErrorCode | None = None,
    ) -> None:
        try:
            self.required(phase, completed_records=completed_records, error_code=error_code)
        except Exception:
            logger.warning("Could not persist client %s progress", phase.value)

    def on_batch_complete(self, path: Path) -> None:
        generated_records = sum(lazy.pq.read_metadata(item).num_rows for item in path.parent.glob("*.parquet"))
        self.best_effort(
            ClientProgressPhase.GENERATING,
            completed_records=min(generated_records, self._context.shard.requested_records),
        )


class ClientWorker:
    """Validate and execute one planned Data Designer shard."""

    def __init__(
        self,
        *,
        data_designer_factory: DataDesignerFactory = DataDesigner,
        clock: Clock | None = None,
    ) -> None:
        self._data_designer_factory = data_designer_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def preflight(
        self,
        plan_path: Path,
        *,
        prepared: PreparedClientEnvironment,
        endpoints: Mapping[str, str],
        plugins: tuple[ClientPluginEntryPoint, ...],
    ) -> ClientEnvironmentManifest:
        """Validate packages, plugins, assets, and config without generation."""
        try:
            context = self._build_context(plan_path, prepared=prepared, endpoints=endpoints)
            progress = _ProgressWriter(context, prepared, self._clock)
            progress.required(ClientProgressPhase.VALIDATING_PLUGINS)
            progress.required(ClientProgressPhase.VALIDATING_CONFIG)
            context.designer.validate(context.builder)
            manifest = ClientEnvironmentManifest.from_prepared(
                prepared,
                created_at=self._clock(),
                outcome=ClientEnvironmentOutcome.READY,
                plugins=plugins,
            )
            replace_private_text(prepared.attempt_dir / "client-environment.json", manifest.serialize_json())
            return manifest
        except ClientWorkerError as error:
            self._write_preflight_failure(plan_path, prepared, endpoints, plugins, error)
            raise
        except Exception as error:
            failure = ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "Data Designer configuration is invalid")
            self._write_preflight_failure(plan_path, prepared, endpoints, plugins, failure)
            raise failure from error

    def run(
        self,
        plan_path: Path,
        *,
        prepared: PreparedClientEnvironment,
        endpoints: Mapping[str, str],
        plugins: tuple[ClientPluginEntryPoint, ...],
    ) -> ClientResult:
        """Invoke the public Data Designer generation contract and persist its result."""
        context: _ExecutionContext | None = None
        progress: _ProgressWriter | None = None
        try:
            context = self._build_context(plan_path, prepared=prepared, endpoints=endpoints)
            progress = _ProgressWriter(context, prepared, self._clock, revision=2)
            self._validate_environment_manifest(context, prepared, plugins)
            progress.required(ClientProgressPhase.GENERATING, completed_records=0)
            self._prepare_dataset_workspace(context, prepared)
            results = self._generate_dataset(context, progress)
            progress.required(ClientProgressPhase.FINALIZING, completed_records=results.actual_num_records)
            try:
                result = self._finalize(context, prepared, results)
            except ClientWorkerError:
                raise
            except Exception as error:
                raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "Data Designer output is invalid") from error
            progress.best_effort(
                ClientProgressPhase.COMPLETE,
                completed_records=cast(int, result.actual_records),
            )
            return result
        except ClientWorkerError as error:
            if context is not None:
                self._write_failure(context, prepared, error, progress=progress)
            raise
        except (KeyboardInterrupt, SystemExit) as error:
            failure = ClientWorkerError(ClientErrorCode.INTERRUPTED, "Data Designer generation was interrupted")
            if context is not None:
                self._write_failure(context, prepared, failure, progress=progress)
            raise failure from error
        except Exception as error:
            failure = ClientWorkerError(ClientErrorCode.GENERATION_FAILED, "Data Designer generation failed")
            if context is not None:
                self._write_failure(context, prepared, failure, progress=progress)
            raise failure from error

    def _build_context(
        self,
        plan_path: Path,
        *,
        prepared: PreparedClientEnvironment,
        endpoints: Mapping[str, str],
    ) -> _ExecutionContext:
        try:
            plan = ResolvedSlurmRunPlan.model_validate_json(
                read_regular_bytes(plan_path, missing_code=ClientErrorCode.INVALID_INPUT)
            )
            shard = next(item for item in plan.shards if item.shard_id == prepared.shard_id)
            self._validate_prepared(plan, shard, prepared)
            lock = ResolvedDependencyLock.model_validate_json(
                read_regular_bytes(
                    Path(plan.client.dependency_lock.path), missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING
                )
            )
            if lock.compute_sha256() != plan.client.dependency_lock.sha256:
                raise ClientWorkerError(ClientErrorCode.DEPENDENCY_DIGEST_MISMATCH, "dependency lock digest differs")
            expected_distributions = tuple(
                sorted(
                    (
                        *lock.image_distributions,
                        *(
                            InstalledDistribution(name=package.name, version=package.version)
                            for package in lock.overlay_packages
                        ),
                    ),
                    key=lambda item: item.name,
                )
            )
            if prepared.installed_distributions != expected_distributions:
                raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "client environment differs from the lock")
            builder_payload = self._load_builder(plan)
            builder = DataDesignerConfigBuilder.from_config(builder_payload)
            providers = self._materialize_model_endpoints(plan, builder, endpoints)
            self._validate_model_references(builder)
            self._materialize_seed(plan, shard, builder)
            mcp_providers = self._materialize_mcp_providers(plan)
            self._validate_assets(plan)
            requested_resume = ResumeMode(plan.invocation.authored.resume)
            dataset_path = self._dataset_path(shard, prepared, requested_resume)
            designer = self._data_designer_factory(
                artifact_path=dataset_path.parent,
                model_providers=providers,
                managed_assets_path=plan.invocation.effective_input_bindings.managed_assets_path,
                mcp_providers=mcp_providers,
                auto_configure_logging=False,
            )
            designer.set_run_config(RunConfig.model_validate(plan.invocation.effective_run_config))
            return _ExecutionContext(plan, shard, builder, designer, requested_resume, dataset_path)
        except ClientWorkerError:
            raise
        except (StopIteration, ValidationError, ValueError, OSError, KeyError, TypeError) as error:
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "Data Designer configuration is invalid") from error

    @staticmethod
    def _validate_prepared(
        plan: ResolvedSlurmRunPlan,
        shard: PlannedShard,
        prepared: PreparedClientEnvironment,
    ) -> None:
        expected_attempt = Path(shard.resume_workspace.path).parent / "attempts" / prepared.attempt_id
        inspection = plan.client.image.inspection_facts
        if (
            plan.run_id != prepared.run_id
            or prepared.attempt_dir != expected_attempt
            or plan.client.dependency_lock != prepared.dependency_lock
            or plan.client.image.sha256 != prepared.client_image_sha256
            or inspection.python_abi != prepared.python_abi
        ):
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "prepared client environment differs from the plan")

    @staticmethod
    def _load_builder(plan: ResolvedSlurmRunPlan) -> dict[str, object]:
        if plan.builder.inline is not None:
            return cast(dict[str, object], plan.builder.inline)
        assert plan.builder.source is not None
        payload = read_regular_bytes(Path(plan.builder.source.path), missing_code=ClientErrorCode.INVALID_INPUT)
        if hashlib.sha256(payload).hexdigest() != plan.builder.source.sha256:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "builder artifact digest differs")
        try:
            loaded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "builder artifact is invalid") from error
        if not isinstance(loaded, dict):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "builder artifact is invalid")
        return cast(dict[str, object], loaded)

    @staticmethod
    def _materialize_model_endpoints(
        plan: ResolvedSlurmRunPlan,
        builder: DataDesignerConfigBuilder,
        endpoints: Mapping[str, str],
    ) -> list[ModelProvider]:
        aliases = tuple(deployment.authored.model_alias for deployment in plan.deployments)
        if set(endpoints) != set(aliases):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "runtime endpoints do not match model aliases")
        configs = {config.alias: config for config in builder.model_configs}
        if set(configs) != set(aliases):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "builder model aliases do not match the plan")
        updated = []
        providers = []
        for deployment, port in zip(plan.deployments, plan.client.ports, strict=True):
            alias = deployment.authored.model_alias
            endpoint = endpoints[alias]
            expected_endpoint = f"http://127.0.0.1:{port.port}/v1"
            parsed = urlsplit(endpoint)
            if endpoint != expected_endpoint or parsed.username is not None or parsed.password is not None:
                raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "runtime model endpoint is invalid")
            provider_name = f"slurm-{alias}"
            model_config = configs[alias]
            inference = model_config.inference_parameters
            concurrency = plan.invocation.authored.model_concurrency.get(alias)
            if concurrency is not None:
                inference = inference.model_copy(update={"max_parallel_requests": concurrency})
            updated.append(
                model_config.model_copy(
                    update={
                        "model": deployment.served_model_name,
                        "provider": provider_name,
                        "inference_parameters": inference,
                    }
                )
            )
            providers.append(ModelProvider(name=provider_name, endpoint=endpoint, provider_type="openai"))
        for model_config in tuple(builder.model_configs):
            builder.delete_model_config(model_config.alias)
        for model_config in updated:
            builder.add_model_config(model_config)
        return providers

    @staticmethod
    def _validate_model_references(builder: DataDesignerConfigBuilder) -> None:
        available = {config.alias for config in builder.model_configs}
        referenced = {alias for column in builder.get_column_configs() for alias in column.get_model_aliases()}
        referenced.update(
            profiler.model_alias for profiler in builder.get_profilers() if hasattr(profiler, "model_alias")
        )
        if not referenced.issubset(available):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "builder references unavailable model aliases")

    @staticmethod
    def _materialize_seed(
        plan: ResolvedSlurmRunPlan,
        shard: PlannedShard,
        builder: DataDesignerConfigBuilder,
    ) -> None:
        seed_path = plan.invocation.effective_input_bindings.seed_path
        if seed_path is None:
            return
        seed = builder.get_seed_config()
        if seed is None or "path" not in type(seed.source).model_fields:
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "seed binding does not match the builder")
        path = Path(seed_path)
        if not path.exists() or not os.access(path, os.R_OK):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "seed input is unavailable")
        if shard.input_partition is None:
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "seed partition artifact is missing")
        payload = read_regular_bytes(Path(shard.input_partition.path), missing_code=ClientErrorCode.INVALID_INPUT)
        if hashlib.sha256(payload).hexdigest() != shard.input_partition.sha256:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "seed partition artifact digest differs")
        source = seed.source.model_copy(update={"path": seed_path})
        builder.with_seed_dataset(
            source,
            sampling_strategy=seed.sampling_strategy,
            selection_strategy=PartitionBlock(index=shard.shard_index, num_partitions=len(plan.shards)),
        )

    @staticmethod
    def _materialize_mcp_providers(
        plan: ResolvedSlurmRunPlan,
    ) -> list[MCPProvider | LocalStdioMCPProvider]:
        providers: list[MCPProvider | LocalStdioMCPProvider] = []
        for provider in plan.invocation.authored.mcp_providers:
            if isinstance(provider, RemoteMCPProviderConfig):
                api_key = _resolve_secret(provider.api_key) if provider.api_key is not None else None
                providers.append(
                    MCPProvider(
                        provider_type=provider.provider_type,
                        name=provider.name,
                        endpoint=provider.endpoint,
                        api_key=api_key,
                    )
                )
            elif isinstance(provider, LocalStdioMCPProviderConfig):
                environment = {
                    name: binding.value if isinstance(binding, LiteralEnvironmentBinding) else _resolve_secret(binding)
                    for name, binding in provider.environment.items()
                }
                providers.append(
                    LocalStdioMCPProvider(
                        name=provider.name,
                        command=provider.command,
                        args=list(provider.args),
                        env=environment,
                    )
                )
        return providers

    @staticmethod
    def _validate_assets(plan: ResolvedSlurmRunPlan) -> None:
        value = plan.invocation.effective_input_bindings.managed_assets_path
        assert value is not None
        path = Path(value)
        if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "managed assets are unavailable")

    @staticmethod
    def _dataset_path(
        shard: PlannedShard,
        prepared: PreparedClientEnvironment,
        resume: ResumeMode,
    ) -> Path:
        resume_path = Path(shard.resume_workspace.path)
        if resume_path.is_symlink():
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "resume workspace is invalid")
        if resume is ResumeMode.ALWAYS and (not resume_path.is_dir() or not any(resume_path.iterdir())):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "required resume workspace is unavailable")
        if resume is ResumeMode.NEVER:
            return prepared.attempt_dir / "dataset"
        if resume_path.exists() and not resume_path.is_dir():
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "resume workspace is invalid")
        return resume_path

    @staticmethod
    def _prepare_dataset_workspace(
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
    ) -> None:
        if context.dataset_path.parent == prepared.attempt_dir:
            ensure_private_directory(context.dataset_path.parent)
        elif not context.dataset_path.parent.is_dir():
            raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "shard dataset workspace is unavailable")
        if context.requested_resume is not ResumeMode.NEVER:
            ensure_private_directory(context.dataset_path)
        if context.requested_resume is ResumeMode.NEVER and context.dataset_path.exists():
            if not context.dataset_path.is_dir() or any(context.dataset_path.iterdir()):
                raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "attempt dataset workspace is not empty")

    @staticmethod
    def _generate_dataset(
        context: _ExecutionContext,
        progress: _ProgressWriter,
    ) -> CreationResults:
        try:
            return cast(
                CreationResults,
                context.designer.create(
                    context.builder,
                    num_records=context.shard.requested_records,
                    dataset_name=context.dataset_path.name,
                    resume=context.requested_resume,
                    artifact_path=context.dataset_path.parent,
                    on_batch_complete=progress.on_batch_complete,
                ),
            )
        except ClientWorkerError:
            raise
        except (KeyboardInterrupt, SystemExit) as error:
            raise ClientWorkerError(ClientErrorCode.INTERRUPTED, "Data Designer generation was interrupted") from error
        except Exception as error:
            raise ClientWorkerError(ClientErrorCode.GENERATION_FAILED, "Data Designer generation failed") from error

    def _finalize(
        self,
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
        results: CreationResults,
    ) -> ClientResult:
        creation = self._validate_creation_result(context, prepared, results)
        dataset_path, exported_path = self._place_dataset(creation, results)
        candidate = self._build_candidate_manifest(context, prepared, creation, dataset_path, exported_path)
        return self._publish_success(context, prepared, creation, candidate)

    @staticmethod
    def _validate_creation_result(
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
        results: CreationResults,
    ) -> _ValidatedCreation:
        requested = context.shard.requested_records
        actual = results.actual_num_records
        if results.requested_num_records != requested or actual > requested:
            raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "Data Designer result counts are invalid")
        if results.early_shutdown is None or results.effective_resume_mode is None:
            raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "Data Designer result metadata is incomplete")
        if results.requested_resume_mode is not context.requested_resume:
            raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "Data Designer resume metadata differs")

        dataset_path = Path(results.dataset_path)
        effective_resume = results.effective_resume_mode
        shared_path = Path(context.shard.resume_workspace.path)
        expected_path = shared_path if effective_resume is ResumeMode.ALWAYS else prepared.attempt_dir / "dataset"
        if (
            not dataset_path.is_absolute()
            or dataset_path != Path(os.path.normpath(dataset_path.as_posix()))
            or not dataset_path.is_dir()
            or dataset_path.is_symlink()
        ):
            raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "Data Designer dataset path is invalid")
        if dataset_path != expected_path and (
            effective_resume is not ResumeMode.NEVER
            or expected_path.exists()
            or not (
                dataset_path.parent == prepared.attempt_dir
                or dataset_path == shared_path
                or (dataset_path.parent == shared_path.parent and dataset_path.name.startswith(f"{shared_path.name}_"))
            )
        ):
            raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "Data Designer dataset path is invalid")
        return _ValidatedCreation(
            requested_records=requested,
            actual_records=actual,
            dataset_path=dataset_path,
            expected_path=expected_path,
            effective_resume=effective_resume,
            early_shutdown=results.early_shutdown,
        )

    @staticmethod
    def _place_dataset(
        creation: _ValidatedCreation,
        results: CreationResults,
    ) -> tuple[Path, Path | None]:
        dataset_path = creation.dataset_path
        exported_path: Path | None = None
        if creation.actual_records:
            exported_path = results.export(dataset_path / "part-00000.parquet", format="parquet")
        if dataset_path != creation.expected_path:
            ensure_private_directory(creation.expected_path.parent)
            dataset_path.rename(creation.expected_path)
            if exported_path is not None:
                exported_path = creation.expected_path / exported_path.relative_to(dataset_path)
            dataset_path = creation.expected_path
        if dataset_path != creation.expected_path:
            raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "Data Designer dataset path is invalid")
        return dataset_path, exported_path

    def _build_candidate_manifest(
        self,
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
        creation: _ValidatedCreation,
        dataset_path: Path,
        exported_path: Path | None,
    ) -> CandidateOutputManifest:
        files: tuple[CandidateOutputFile, ...] = ()
        schema_digest = context.plan.builder.content_sha256
        if exported_path is not None:
            metadata = lazy.pq.read_metadata(exported_path)
            if metadata.num_rows != creation.actual_records:
                raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "exported record count differs")
            schema_digest = hashlib.sha256(lazy.pq.read_schema(exported_path).serialize().to_pybytes()).hexdigest()
            files = (
                CandidateOutputFile(
                    relative_path=exported_path.relative_to(dataset_path).as_posix(),
                    sha256=compute_file_sha256(exported_path, missing_code=ClientErrorCode.OUTPUT_INVALID),
                    byte_size=exported_path.stat().st_size,
                    record_count=creation.actual_records,
                ),
            )
        created_at = self._clock()
        provenance_digest = compute_canonical_json_sha256(
            {
                "builder_sha256": context.plan.builder.content_sha256,
                "client_image_sha256": prepared.client_image_sha256,
                "dependency_lock_sha256": prepared.dependency_lock.sha256,
                "files": [file.model_dump(mode="json") for file in files],
                "attempt_id": prepared.attempt_id,
                "resolved_plan_sha256": context.plan.compute_sha256(),
                "run_id": context.plan.run_id,
                "shard_id": context.shard.shard_id,
            }
        )
        return CandidateOutputManifest(
            schema_version=1,
            run_id=context.plan.run_id,
            shard_id=context.shard.shard_id,
            attempt_id=prepared.attempt_id,
            attempt_ordinal=int(prepared.attempt_id.removeprefix("attempt-")),
            created_at=created_at,
            dataset_path=dataset_path.as_posix(),
            requested_records=creation.requested_records,
            actual_records=creation.actual_records,
            outcome=(
                CandidateOutcome.EMPTY
                if creation.actual_records == 0
                else CandidateOutcome.COMPLETE
                if creation.actual_records == creation.requested_records
                else CandidateOutcome.PARTIAL
            ),
            files=files,
            dataset_schema_digest=schema_digest,
            provenance_digest=provenance_digest,
        )

    def _publish_success(
        self,
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
        creation: _ValidatedCreation,
        candidate: CandidateOutputManifest,
    ) -> ClientResult:
        candidate_path = prepared.attempt_dir / "output-manifest.json"
        publish_private_text(candidate_path, candidate.serialize_json())
        result = ClientResult(
            schema_version=1,
            run_id=context.plan.run_id,
            shard_id=context.shard.shard_id,
            attempt_id=prepared.attempt_id,
            completed_at=self._clock(),
            requested_records=creation.requested_records,
            actual_records=creation.actual_records,
            outcome=(
                ClientOutcome.COMPLETE
                if creation.actual_records == creation.requested_records
                else ClientOutcome.PARTIAL
            ),
            dataset_path=candidate.dataset_path,
            early_shutdown=creation.early_shutdown,
            requested_resume_mode=context.requested_resume.value,
            effective_resume_mode=creation.effective_resume.value,
            candidate_output_manifest=ArtifactReference(
                path=candidate_path.as_posix(), sha256=candidate.compute_sha256()
            ),
        )
        publish_private_text(prepared.attempt_dir / "client-result.json", result.serialize_json())
        return result

    def _validate_environment_manifest(
        self,
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
        plugins: tuple[ClientPluginEntryPoint, ...],
    ) -> None:
        try:
            manifest = ClientEnvironmentManifest.model_validate_json(
                read_regular_bytes(
                    prepared.attempt_dir / "client-environment.json",
                    missing_code=ClientErrorCode.CONFIG_INVALID,
                )
            )
        except ValidationError as error:
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "client environment manifest is invalid") from error
        if (
            manifest.outcome is not ClientEnvironmentOutcome.READY
            or manifest.run_id != context.plan.run_id
            or manifest.shard_id != context.shard.shard_id
            or manifest.attempt_id != prepared.attempt_id
            or manifest.dependency_lock != prepared.dependency_lock
            or manifest.client_image_sha256 != prepared.client_image_sha256
            or manifest.python_abi != prepared.python_abi
            or manifest.overlay_path != prepared.overlay_path.as_posix()
            or manifest.installed_distributions != prepared.installed_distributions
            or manifest.plugins != plugins
        ):
            raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "client environment manifest differs")

    def _write_preflight_failure(
        self,
        plan_path: Path,
        prepared: PreparedClientEnvironment,
        endpoints: Mapping[str, str],
        plugins: tuple[ClientPluginEntryPoint, ...],
        error: ClientWorkerError,
    ) -> None:
        manifest = ClientEnvironmentManifest.from_prepared(
            prepared,
            created_at=self._clock(),
            outcome=ClientEnvironmentOutcome.FAILED,
            plugins=plugins,
            error_code=error.code,
            redacted_message=error.redacted_message,
        )
        replace_private_text(prepared.attempt_dir / "client-environment.json", manifest.serialize_json())
        try:
            context = self._build_context(plan_path, prepared=prepared, endpoints=endpoints)
        except ClientWorkerError:
            return
        self._write_failure(context, prepared, error, progress=None)

    def _write_failure(
        self,
        context: _ExecutionContext,
        prepared: PreparedClientEnvironment,
        error: ClientWorkerError,
        *,
        progress: _ProgressWriter | None,
    ) -> None:
        result = ClientResult(
            schema_version=1,
            run_id=context.plan.run_id,
            shard_id=context.shard.shard_id,
            attempt_id=prepared.attempt_id,
            completed_at=self._clock(),
            requested_records=context.shard.requested_records,
            actual_records=None,
            outcome=ClientOutcome.FAILED,
            requested_resume_mode=context.requested_resume.value,
            error_code=error.code.value,
            redacted_message=error.redacted_message,
        )
        publish_private_text(prepared.attempt_dir / "client-result.json", result.serialize_json())
        progress = progress or _ProgressWriter(context, prepared, self._clock, revision=2)
        progress.best_effort(ClientProgressPhase.FAILED, error_code=error.code)


def _resolve_secret(reference: SecretRef) -> str:
    try:
        return os.environ[reference.environment]
    except KeyError as error:
        raise ClientWorkerError(ClientErrorCode.CONFIG_INVALID, "required client secret is unavailable") from error
