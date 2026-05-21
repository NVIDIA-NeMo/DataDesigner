# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from data_designer.engine.models.resources import ProviderModelKey


class RequestDomain(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    HEALTHCHECK = "healthcheck"


@dataclass(frozen=True, order=True)
class RequestResourceKey:
    provider_name: str
    model_id: str
    domain: RequestDomain

    @property
    def provider_model_key(self) -> ProviderModelKey:
        return ProviderModelKey(self.provider_name, self.model_id)


@dataclass(frozen=True)
class RequestGroupSpec:
    key: RequestResourceKey
    weight: float = 1.0


@dataclass(frozen=True)
class RequestEventContext:
    captured_correlation: object | None = None
    task_execution_id: str | None = None
    request_attempt_id: str | None = None


@dataclass(frozen=True)
class RequestAdmissionItem:
    resource: RequestResourceKey
    group: RequestGroupSpec
    queue_wait_timeout_seconds: float | None = None
    event_context: RequestEventContext | None = None
