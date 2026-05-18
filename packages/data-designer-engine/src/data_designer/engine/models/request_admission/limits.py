# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveRequestLimitState:
    current_limit: int
    in_flight: int = 0
    blocked_until: float = 0.0
    success_streak: int = 0
    waiters: int = 0
    rate_limit_ceiling: int = 0
    consecutive_rate_limits: int = 0
    active_lease_count: int = 0
    last_outcome: str | None = None
