# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RequestReleaseOutcome:
    kind: Literal[
        "success",
        "rate_limited",
        "provider_failure",
        "provider_timeout",
        "local_cancelled",
        "local_timeout",
        "unexpected_exception",
    ]
    retry_after_seconds: float | None = None
    provider_status: int | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseResult:
    released: bool
    reason: Literal["released", "duplicate", "stale_lease", "wrong_controller_generation", "unknown_lease"]
    diagnostics: Mapping[str, object] = field(default_factory=dict)
