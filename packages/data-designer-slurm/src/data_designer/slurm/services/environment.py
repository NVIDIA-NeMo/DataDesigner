# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-only credential transport shared by initial submissions and retries."""

from __future__ import annotations

from collections.abc import Mapping

from data_designer.slurm.config import collect_secret_environment_names
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services.errors import SlurmServiceError, SlurmServiceErrorCode, SlurmServiceOperation


def build_run_export_environment(
    plan: ResolvedSlurmRunPlan,
    source_environment: Mapping[str, str],
    operation: SlurmServiceOperation,
) -> dict[str, str]:
    """Forward only required credentials without serializing their values."""
    environment = {"SLURM_EXPORT_ENV": "ALL"}
    deployment_type = source_environment.get("NEMO_DEPLOYMENT_TYPE", "slurm")
    if type(deployment_type) is not str or "\0" in deployment_type:
        raise SlurmServiceError(
            SlurmServiceErrorCode.INVALID_REQUEST,
            operation,
            "NEMO_DEPLOYMENT_TYPE is invalid",
        )
    environment["NEMO_DEPLOYMENT_TYPE"] = deployment_type
    if "SLURM_CONF" in source_environment:
        slurm_conf = source_environment["SLURM_CONF"]
        if type(slurm_conf) is not str or "\0" in slurm_conf:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "SLURM_CONF is invalid",
            )
        environment["SLURM_CONF"] = slurm_conf
    for name in collect_secret_environment_names(plan):
        try:
            value = source_environment[name]
        except KeyError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                f"required secret environment {name!r} is unavailable",
            ) from None
        if type(value) is not str or "\0" in value:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                f"required secret environment {name!r} is invalid",
            )
        environment[name] = value
    return environment
