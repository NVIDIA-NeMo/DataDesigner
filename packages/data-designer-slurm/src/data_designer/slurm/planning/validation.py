# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import posixpath

from pydantic import JsonValue

from data_designer.config import RunConfig
from data_designer.slurm.config.images import ClientImageInspection
from data_designer.slurm.config.run import DataDesignerSlurmConfig
from data_designer.slurm.planning.builder_identity import get_persisted_builder_identity
from data_designer.slurm.planning.errors import SlurmPlanContractError
from data_designer.slurm.planning.models import (
    ResolvedDependencyLock,
    ResolvedSlurmRunPlan,
)


def validate_resolved_plan(
    authored: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    plan: ResolvedSlurmRunPlan,
    *,
    builder_payload: dict[str, JsonValue] | None = None,
) -> ResolvedSlurmRunPlan:
    """Validate cross-record identities and digests for one resolved plan."""
    _require(
        plan.authored_config.sha256 == authored.compute_sha256(),
        "authored config digest does not match the resolved plan",
    )
    _require(plan.invocation.authored == authored.invocation, "resolved invocation does not match authored input")
    explicit_run_config = RunConfig.model_validate(authored.invocation.run_config).model_dump(
        mode="json",
        exclude_unset=True,
    )
    _require_json_subset(plan.invocation.effective_run_config, explicit_run_config, path="run_config")
    _require(plan.client.authored == authored.client, "resolved client does not match authored input")
    _require(
        tuple(deployment.authored for deployment in plan.deployments) == tuple(authored.deployments),
        "resolved deployments do not match authored order and values",
    )
    _require(plan.array_tasks == authored.array_tasks, "resolved array task policy does not match authored input")

    if authored.builder.inline is not None:
        _require(builder_payload is None, "inline builder input must not retain a separate payload")
        _require(
            plan.builder.inline == authored.builder.inline, "resolved inline builder does not match authored input"
        )
        model_aliases, digest = get_persisted_builder_identity(authored.builder.inline)
        _require(plan.builder.model_aliases == model_aliases, "resolved model aliases do not match inline builder")
        _require(plan.builder.content_sha256 == digest, "resolved builder digest does not match inline builder")
    else:
        _require(
            plan.builder.authored_source == authored.builder.source,
            "resolved builder source does not match authored input",
        )
        if builder_payload is None:
            raise SlurmPlanContractError("sourced builder validation requires its resolved payload")
        model_aliases, digest = get_persisted_builder_identity(builder_payload)
        _require(plan.builder.source is not None, "resolved builder source artifact is missing")
        assert plan.builder.source is not None
        expected_path = posixpath.join(posixpath.dirname(plan.authored_config.path), "builder-config.json")
        _require(plan.builder.source.path == expected_path, "resolved builder artifact path does not match the run")
        _require(plan.builder.model_aliases == model_aliases, "resolved model aliases do not match builder source")
        _require(plan.builder.source.sha256 == digest, "resolved builder digest does not match builder source")
        _require(plan.builder.content_sha256 == digest, "resolved builder digest does not match builder source")

    expected_account = authored.submission.account or plan.selected_profile.profile.scheduler.account
    expected_partition = authored.submission.partition or plan.selected_profile.profile.scheduler.partition
    _require(plan.submission.account == expected_account, "resolved account does not match authored/profile input")
    _require(
        plan.submission.partition == expected_partition, "resolved partition does not match authored/profile input"
    )
    _require(
        plan.submission.job_name == authored.submission.job_name, "resolved job name does not match authored input"
    )
    _require(
        plan.submission.time_limit == authored.submission.time_limit,
        "resolved time limit does not match authored input",
    )
    _require(plan.submission.comment == authored.submission.comment, "resolved comment does not match authored input")

    _require(plan.output.format == authored.output.format, "resolved output format does not match authored input")
    _require(
        plan.output.partitions == authored.output.partitions, "resolved output partitions do not match authored input"
    )
    _require(
        plan.output.require_exact_record_count == authored.output.require_exact_record_count,
        "resolved exact-record policy does not match authored input",
    )
    if authored.output.root is not None:
        _require(plan.output.root == authored.output.root, "resolved output root does not match authored input")

    _require(
        plan.client.dependency_lock.sha256 == dependency_lock.compute_sha256(),
        "dependency lock digest does not match the resolved plan",
    )
    _require(
        dependency_lock.client_image_sha256 == plan.client.image.sha256,
        "dependency lock client image digest does not match the resolved client image",
    )
    inspection = plan.client.image.inspection.inspection
    _require(isinstance(inspection, ClientImageInspection), "resolved client image lacks client inspection facts")
    _require(
        dependency_lock.python_abi == inspection.python_abi, "dependency lock Python ABI does not match client image"
    )
    _require(
        dependency_lock.image_distributions == inspection.distributions,
        "dependency lock image inventory does not match client image inspection",
    )
    authored_requirements = authored.client.dependencies.requirements
    if authored_requirements is not None:
        _require(
            dependency_lock.authored_source is None and dependency_lock.source is None,
            "dependency lock source is present for authored requirements",
        )
        _require(
            dependency_lock.authored_requirements == tuple(authored_requirements),
            "dependency lock requirements do not match authored requirements",
        )
    else:
        _require(
            dependency_lock.authored_source == authored.client.dependencies.lock_file
            and dependency_lock.source is not None,
            "dependency lock source does not match the authored lock file",
        )
    return plan


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlurmPlanContractError(message)


def _require_json_subset(actual: JsonValue, expected: JsonValue, *, path: str) -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"explicit {path} value does not match the resolved plan")
        for key, value in expected.items():
            _require(key in actual, f"explicit {path}.{key} value is missing from the resolved plan")
            _require_json_subset(actual[key], value, path=f"{path}.{key}")
        return
    _require(actual == expected, f"explicit {path} value does not match the resolved plan")
