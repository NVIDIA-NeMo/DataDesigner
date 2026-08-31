# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal pure authored-configuration resolution for Slurm planning."""

from __future__ import annotations

import posixpath
from typing import Annotated

from pydantic import JsonValue, PositiveInt, StringConstraints, TypeAdapter, ValidationError

from data_designer.config import RunConfig
from data_designer.slurm._errors import format_validation_error
from data_designer.slurm.config.images import ClientImageInspection, ImageKind
from data_designer.slurm.config.profiles import SelectedSlurmProfile
from data_designer.slurm.config.run import BuilderInput, DataDesignerSlurmConfig, InputBindings
from data_designer.slurm.contracts import (
    ArtifactReference,
    ContractValue,
    Identifier,
    compute_serialized_json_sha256,
    derive_managed_assets_path,
    is_path_below,
    paths_overlap,
)
from data_designer.slurm.planning.builder_identity import (
    get_declared_model_aliases,
    get_persisted_builder_identity,
)
from data_designer.slurm.planning.errors import SlurmConfigResolutionError
from data_designer.slurm.planning.models import (
    ResolvedBuilderInput,
    ResolvedDependencyLock,
    ResolvedImage,
    ResolvedInvocation,
    ResolvedOutput,
    ResolvedSubmission,
)

__all__: list[str] = []

_SHARDABLE_COLUMN_TYPES = frozenset(
    {
        "embedding",
        "expression",
        "llm-code",
        "llm-judge",
        "llm-structured",
        "llm-text",
        "sampler",
        "seed-dataset",
        "validation",
    }
)
_COMPATIBILITY_RUN_DEFAULTS: dict[str, JsonValue] = {
    "buffer_size": 16384,
    "disable_early_shutdown": True,
    "display_tui": False,
    "max_conversation_correction_steps": 0,
    "max_conversation_restarts": 0,
    "non_inference_max_parallel_workers": 4,
    "otel_metrics_port": None,
    "shutdown_error_rate": 1.0,
}
_RUN_ID_ADAPTER = TypeAdapter(Identifier)
_PACKAGE_VERSION_ADAPTER = TypeAdapter(Annotated[str, StringConstraints(min_length=1, max_length=128)])
_RESOLUTION_VALIDATION_MODELS = (
    RunConfig,
    ArtifactReference,
    BuilderInput,
    InputBindings,
    ResolvedBuilderInput,
    ResolvedInvocation,
    ResolvedOutput,
    ResolvedSubmission,
)


class EffectiveDataDesignerSlurmConfig(ContractValue):
    """Fully materialized, side-effect-free input to the plan compiler."""

    run_id: Identifier
    package_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    authored: DataDesignerSlurmConfig
    selected_profile: SelectedSlurmProfile
    resolved_gpus_per_node: PositiveInt
    builder: ResolvedBuilderInput
    builder_payload: dict[str, JsonValue] | None = None
    invocation: ResolvedInvocation
    client_image: ResolvedImage
    deployment_images: tuple[ResolvedImage, ...]
    dependency_lock: ResolvedDependencyLock
    submission: ResolvedSubmission
    output: ResolvedOutput
    runtime_bundle: ArtifactReference


def resolve_slurm_config(
    authored: DataDesignerSlurmConfig,
    *,
    selected_profile: SelectedSlurmProfile,
    client_image: ResolvedImage,
    deployment_images: tuple[ResolvedImage, ...],
    dependency_lock: ResolvedDependencyLock,
    runtime_bundle: ArtifactReference,
    run_id: str,
    package_version: str,
    resolved_gpus_per_node: int | None = None,
    builder_payload: dict[str, JsonValue] | None = None,
) -> EffectiveDataDesignerSlurmConfig:
    """Materialize all non-secret defaults from explicitly supplied resolved inputs."""
    try:
        run_id = _RUN_ID_ADAPTER.validate_python(run_id, strict=True)
        package_version = _PACKAGE_VERSION_ADAPTER.validate_python(package_version, strict=True)
        gpus_per_node = _resolve_gpu_count(selected_profile, resolved_gpus_per_node)
        workspace_root = selected_profile.profile.workspace_root
        run_root = posixpath.join(workspace_root, "runs", run_id)
        builder, resolved_builder_payload = _resolve_builder(
            authored,
            run_root=run_root,
            builder_payload=builder_payload,
        )
        effective = EffectiveDataDesignerSlurmConfig(
            run_id=run_id,
            package_version=package_version,
            authored=authored,
            selected_profile=selected_profile,
            resolved_gpus_per_node=gpus_per_node,
            builder=builder,
            builder_payload=resolved_builder_payload,
            invocation=_materialize_invocation(authored, workspace_root),
            client_image=client_image,
            deployment_images=deployment_images,
            dependency_lock=dependency_lock,
            submission=_materialize_submission(authored, selected_profile),
            output=_materialize_output(authored, run_root),
            runtime_bundle=runtime_bundle,
        )
        return validate_effective_slurm_config(effective)
    except SlurmConfigResolutionError:
        raise
    except ValidationError as error:
        message = format_validation_error(
            error,
            subject="Slurm configuration resolution",
            models=(*_RESOLUTION_VALIDATION_MODELS, EffectiveDataDesignerSlurmConfig),
        )
        raise SlurmConfigResolutionError(message) from None
    except ValueError as error:
        raise SlurmConfigResolutionError(str(error)) from None


def validate_effective_slurm_config(
    effective: EffectiveDataDesignerSlurmConfig,
) -> EffectiveDataDesignerSlurmConfig:
    """Validate one fully materialized compiler input."""
    authored = effective.authored
    profile = effective.selected_profile.profile
    workspace_root = profile.workspace_root
    run_root = posixpath.join(workspace_root, "runs", effective.run_id)
    if profile.gpus_per_node != "auto" and profile.gpus_per_node != effective.resolved_gpus_per_node:
        raise SlurmConfigResolutionError("resolved GPU count does not match the selected profile")
    if profile.gpu_request_mode == "visible" and profile.scheduler.mem_per_gpu is not None:
        raise SlurmConfigResolutionError("mem_per_gpu requires GRES GPU request mode")

    if authored.builder.inline is not None:
        if effective.builder_payload is not None:
            raise SlurmConfigResolutionError("inline builder input must not provide a separate payload")
    else:
        if effective.builder_payload is None:
            raise SlurmConfigResolutionError("sourced builder input requires its resolved payload")
        BuilderInput(inline=effective.builder_payload)
    expected_invocation = _materialize_invocation(authored, workspace_root)
    if effective.invocation != expected_invocation:
        raise SlurmConfigResolutionError("resolved invocation does not match the authored invocation")
    expected_submission = _materialize_submission(authored, effective.selected_profile)
    if effective.submission != expected_submission:
        raise SlurmConfigResolutionError("resolved submission does not match the authored and profile input")
    expected_output = _materialize_output(authored, run_root)
    if effective.output != expected_output:
        raise SlurmConfigResolutionError("resolved output does not match the authored output")

    _validate_dependency_resolution(authored, effective.client_image, effective.dependency_lock)
    _validate_resolved_images(authored, effective.client_image, effective.deployment_images)
    _validate_sharding_constraints(authored, builder_payload=effective.builder_payload)
    managed_assets_path = effective.invocation.effective_input_bindings.managed_assets_path
    assert managed_assets_path is not None
    _validate_managed_assets_path(managed_assets_path, workspace_root)
    _validate_output_destination(
        effective.output.root,
        workspace_root,
        run_root,
        managed_assets_path=managed_assets_path,
    )
    if effective.output.partitions > authored.invocation.num_records:
        raise SlurmConfigResolutionError("output partitions must not exceed requested records")
    runtime_root = posixpath.join(workspace_root, "runtime")
    runtime_name = posixpath.basename(effective.runtime_bundle.path)
    if (
        not is_path_below(effective.runtime_bundle.path, runtime_root)
        or runtime_name != f"{effective.runtime_bundle.sha256}.tar.gz"
    ):
        raise SlurmConfigResolutionError(
            "runtime bundle must be a content-addressed tar archive below the selected workspace runtime root"
        )
    return effective


def _resolve_gpu_count(selected: SelectedSlurmProfile, resolved: int | None) -> int:
    configured = selected.profile.gpus_per_node
    if configured == "auto":
        if type(resolved) is not int or resolved <= 0:
            raise SlurmConfigResolutionError("auto gpus_per_node requires one resolved positive integer")
        return resolved
    if resolved is not None and resolved != configured:
        raise SlurmConfigResolutionError("resolved GPU count conflicts with the selected profile")
    return configured


def _resolve_builder(
    authored: DataDesignerSlurmConfig,
    *,
    run_root: str,
    builder_payload: dict[str, JsonValue] | None,
) -> tuple[ResolvedBuilderInput, dict[str, JsonValue] | None]:
    if authored.builder.inline is not None:
        if builder_payload is not None:
            raise SlurmConfigResolutionError("inline builder input must not provide a separate payload")
        return (
            ResolvedBuilderInput(
                inline=authored.builder.inline,
                content_sha256=compute_serialized_json_sha256(authored.builder.inline),
                model_aliases=get_declared_model_aliases(authored.builder.inline),
            ),
            None,
        )
    if builder_payload is None:
        raise SlurmConfigResolutionError("sourced builder input requires its resolved payload")
    validated_payload = BuilderInput(inline=builder_payload).inline
    assert validated_payload is not None
    aliases, digest = get_persisted_builder_identity(validated_payload)
    source = ArtifactReference(
        path=posixpath.join(run_root, "builder-config.json"),
        sha256=digest,
    )
    return (
        ResolvedBuilderInput(
            authored_source=authored.builder.source,
            source=source,
            content_sha256=source.sha256,
            model_aliases=aliases,
        ),
        validated_payload,
    )


def _materialize_invocation(authored: DataDesignerSlurmConfig, workspace_root: str) -> ResolvedInvocation:
    input_bindings = authored.invocation.input_bindings
    return ResolvedInvocation(
        authored=authored.invocation,
        effective_input_bindings=InputBindings(
            seed_path=input_bindings.seed_path,
            managed_assets_path=input_bindings.managed_assets_path or derive_managed_assets_path(workspace_root),
        ),
        effective_run_config=_materialize_run_config(authored),
    )


def _materialize_submission(
    authored: DataDesignerSlurmConfig,
    selected_profile: SelectedSlurmProfile,
) -> ResolvedSubmission:
    return ResolvedSubmission(
        account=authored.submission.account or selected_profile.profile.scheduler.account,
        partition=authored.submission.partition or selected_profile.profile.scheduler.partition,
        job_name=authored.submission.job_name,
        time_limit=authored.submission.time_limit,
        comment=authored.submission.comment,
    )


def _materialize_output(authored: DataDesignerSlurmConfig, run_root: str) -> ResolvedOutput:
    return ResolvedOutput(
        root=authored.output.root or posixpath.join(run_root, "output"),
        format=authored.output.format,
        partitions=authored.output.partitions,
        require_exact_record_count=authored.output.require_exact_record_count,
    )


def _materialize_run_config(authored: DataDesignerSlurmConfig) -> dict[str, JsonValue]:
    values = dict(authored.invocation.run_config)
    authored_early_shutdown = {"disable_early_shutdown", "shutdown_error_rate", "shutdown_error_window"}.intersection(
        values
    )
    for name, value in _COMPATIBILITY_RUN_DEFAULTS.items():
        if authored_early_shutdown and name in {"disable_early_shutdown", "shutdown_error_rate"}:
            continue
        values.setdefault(name, value)
    return RunConfig.model_validate(values).model_dump(mode="json")


def _validate_sharding_constraints(
    authored: DataDesignerSlurmConfig,
    *,
    builder_payload: dict[str, JsonValue] | None,
) -> None:
    if authored.array_tasks.count == 1:
        return
    if authored.output.format != "parquet":
        raise SlurmConfigResolutionError("multi-shard runs require parquet output")

    payload = authored.builder.inline if authored.builder.inline is not None else builder_payload
    assert payload is not None
    data_designer = payload.get("data_designer", payload)
    if not isinstance(data_designer, dict):
        raise SlurmConfigResolutionError("builder data_designer value must be an object")
    if data_designer.get("processors"):
        raise SlurmConfigResolutionError("multi-shard runs do not support global processors")
    if data_designer.get("profilers"):
        raise SlurmConfigResolutionError("multi-shard runs do not support global profilers")

    seed_config = data_designer.get("seed_config")
    if isinstance(seed_config, dict):
        if seed_config.get("sampling_strategy") == "shuffle":
            raise SlurmConfigResolutionError("multi-shard runs do not support shuffled seed input")
        if seed_config.get("selection_strategy") is not None:
            raise SlurmConfigResolutionError("multi-shard runs do not support authored seed selection strategies")
        if authored.invocation.input_bindings.seed_path is None:
            raise SlurmConfigResolutionError("multi-shard seed input requires a typed seed_path binding")

    columns = data_designer.get("columns", [])
    if not isinstance(columns, list):
        raise SlurmConfigResolutionError("builder columns must be a list")
    for column in columns:
        if not isinstance(column, dict) or not isinstance(column.get("column_type"), str):
            raise SlurmConfigResolutionError("multi-shard runs require known column semantics")
        column_type = column["column_type"]
        if column_type == "image":
            raise SlurmConfigResolutionError("multi-shard runs do not support media output columns")
        if column_type not in _SHARDABLE_COLUMN_TYPES:
            raise SlurmConfigResolutionError(
                "multi-shard runs do not support custom, plugin, or unknown column semantics"
            )
        if column_type == "validation" and column.get("validator_type") == "local_callable":
            raise SlurmConfigResolutionError("multi-shard runs do not support local callable validators")


def _validate_dependency_resolution(
    authored: DataDesignerSlurmConfig,
    client_image: ResolvedImage,
    dependency_lock: ResolvedDependencyLock,
) -> None:
    inspection = client_image.inspection.inspection
    if not isinstance(inspection, ClientImageInspection):
        raise SlurmConfigResolutionError("resolved client image lacks dependency inspection facts")
    if dependency_lock.client_image_sha256 != client_image.sha256:
        raise SlurmConfigResolutionError("dependency lock does not match the resolved client image")
    if dependency_lock.python_abi != inspection.python_abi:
        raise SlurmConfigResolutionError("dependency lock Python ABI does not match the client image")
    if dependency_lock.image_distributions != inspection.distributions:
        raise SlurmConfigResolutionError("dependency lock inventory does not match the client image")
    requirements = authored.client.dependencies.requirements
    if requirements is not None:
        if dependency_lock.authored_source is not None or dependency_lock.source is not None:
            raise SlurmConfigResolutionError("inline requirements cannot resolve from an authored lock file")
        if dependency_lock.authored_requirements != tuple(requirements):
            raise SlurmConfigResolutionError("dependency lock requirements do not match authored requirements")
    elif dependency_lock.authored_source != authored.client.dependencies.lock_file or dependency_lock.source is None:
        raise SlurmConfigResolutionError("dependency lock source does not match the authored lock file")


def _validate_resolved_images(
    authored: DataDesignerSlurmConfig,
    client_image: ResolvedImage,
    deployment_images: tuple[ResolvedImage, ...],
) -> None:
    _validate_resolved_image_identity(client_image)
    if client_image.kind is not ImageKind.CLIENT:
        raise SlurmConfigResolutionError("resolved client image must contain client inspection facts")
    if client_image.authored_ref != authored.client.image:
        raise SlurmConfigResolutionError("resolved client image does not match the authored reference")
    if len(deployment_images) != len(authored.deployments):
        raise SlurmConfigResolutionError("resolved serving images must match the authored deployment count")
    for deployment, image in zip(authored.deployments, deployment_images, strict=True):
        _validate_resolved_image_identity(image)
        if image.kind is not ImageKind.SERVING:
            raise SlurmConfigResolutionError("resolved deployment image must contain serving inspection facts")
        if image.authored_ref != deployment.server.image:
            raise SlurmConfigResolutionError("resolved deployment image does not match the authored reference")


def _validate_resolved_image_identity(image: ResolvedImage) -> None:
    if image.sha256 != image.inspection.sqsh_sha256:
        raise SlurmConfigResolutionError("resolved image digest does not match its inspection record")
    if image.authored_ref.path is not None and image.path != image.authored_ref.path:
        raise SlurmConfigResolutionError("resolved image path does not match the authored path")


def _validate_output_destination(
    output_root: str,
    workspace_root: str,
    run_root: str,
    *,
    managed_assets_path: str,
) -> None:
    if not is_path_below(output_root, workspace_root):
        raise SlurmConfigResolutionError("output root must be below the selected workspace_root")
    reserved = tuple(posixpath.join(workspace_root, name) for name in ("images", "runtime", "benchmarks"))
    if any(paths_overlap(output_root, path) for path in reserved):
        raise SlurmConfigResolutionError("output root must not overlap package-managed workspace state")
    if paths_overlap(output_root, managed_assets_path):
        raise SlurmConfigResolutionError("output root must not overlap managed assets")
    runs_root = posixpath.join(workspace_root, "runs")
    run_output_root = posixpath.join(run_root, "output")
    if paths_overlap(output_root, runs_root) and not (
        output_root == run_output_root or is_path_below(output_root, run_output_root)
    ):
        raise SlurmConfigResolutionError("output root must not overlap another package-managed run")


def _validate_managed_assets_path(managed_assets_path: str, workspace_root: str) -> None:
    workspace_state = tuple(
        posixpath.join(workspace_root, name) for name in ("images", "runtime", "benchmarks", "runs")
    )
    if any(paths_overlap(managed_assets_path, path) for path in workspace_state):
        raise SlurmConfigResolutionError("managed_assets_path must not overlap package-managed workspace state")
