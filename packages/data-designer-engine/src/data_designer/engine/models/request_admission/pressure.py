# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.resources import RequestDomain, RequestResourceKey
from data_designer.engine.models.resources import ProviderModelKey


@dataclass(frozen=True)
class RequestPressureSnapshot:
    captured_at: float
    sequence: int
    resource: RequestResourceKey
    effective_max: int
    current_limit: int
    in_flight_count: int
    active_lease_count: int
    waiters: int
    blocked_until_monotonic: float | None
    cooldown_remaining_seconds: float
    rate_limit_ceiling: int
    consecutive_rate_limits: int
    last_outcome: str | None
    leak_diagnostics: Mapping[str, int]


@dataclass(frozen=True)
class ProviderModelPressureSnapshot:
    captured_at: float
    sequence: int
    provider_model: ProviderModelKey
    static_cap: int
    aggregate_in_flight: int
    aggregate_active_lease_count: int
    aliases: tuple[str, ...]
    raw_caps: Mapping[str, int | None]
    domains: Mapping[RequestDomain, int]


class RequestPressureSnapshotProvider(Protocol):
    @property
    def config(self) -> RequestAdmissionConfig | None: ...

    def snapshot(self, resource: RequestResourceKey) -> RequestPressureSnapshot | None: ...

    def snapshots(self) -> Mapping[RequestResourceKey, RequestPressureSnapshot]: ...

    def global_snapshot(self, provider: str, model: str) -> ProviderModelPressureSnapshot | None: ...

    def global_snapshots(self) -> Mapping[ProviderModelKey, ProviderModelPressureSnapshot]: ...
