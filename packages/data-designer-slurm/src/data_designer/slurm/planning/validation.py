# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import JsonValue

from data_designer.slurm.config.images import ClientImageInspection
from data_designer.slurm.config.run import DataDesignerSlurmConfig
from data_designer.slurm.planning.models import (
    ResolvedDependencyLock,
    ResolvedSlurmRunPlan,
    _extract_builder_aliases,
)


class PlanContractError(ValueError):
    """Raised when a resolved plan does not match its authored inputs."""


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
    _require(plan.client.authored == authored.client, "resolved client does not match authored input")
    _require(
        tuple(deployment.authored for deployment in plan.deployments) == tuple(authored.deployments),
        "resolved deployments do not match authored order and values",
    )
    _require(plan.array_tasks == authored.array_tasks, "resolved array task policy does not match authored input")

    if authored.builder.inline is not None:
        _require(
            plan.builder.inline == authored.builder.inline, "resolved inline builder does not match authored input"
        )
    else:
        _require(
            plan.builder.authored_source == authored.builder.source,
            "resolved builder source does not match authored input",
        )
        if builder_payload is None:
            raise PlanContractError("sourced builder validation requires its resolved payload")
        model_aliases, referenced_aliases = _extract_builder_aliases(builder_payload)
        _require(plan.builder.model_aliases == model_aliases, "resolved model aliases do not match builder source")
        _require(
            plan.builder.referenced_model_aliases == referenced_aliases,
            "resolved referenced aliases do not match builder source",
        )

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
        raise PlanContractError(message)
