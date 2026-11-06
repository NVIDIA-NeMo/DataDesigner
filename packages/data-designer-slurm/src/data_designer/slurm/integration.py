# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public integration validation across Slurm plan and runtime-state contracts."""

from __future__ import annotations

from data_designer.slurm.state.plan_validation import PersistedPlanStateValidator, PlanStateContractError

IntegrationContractError = PlanStateContractError


class PlanStateValidator(PersistedPlanStateValidator):
    """Public compatibility facade for persisted-plan validation."""


__all__ = [
    "IntegrationContractError",
    "PlanStateValidator",
]
