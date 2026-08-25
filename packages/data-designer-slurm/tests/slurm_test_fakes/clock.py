# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class FakeClock:
    """Clock advanced only by explicit test input."""

    current_time: datetime
    monotonic_time: float = 0.0
    sleep_calls: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.current_time.tzinfo is None or self.current_time.utcoffset() != timedelta(0):
            raise ValueError("current_time must be in UTC")
        if self.monotonic_time < 0:
            raise ValueError("monotonic_time must not be negative")

    def now(self) -> datetime:
        """Return the controlled wall-clock time."""
        return self.current_time

    def monotonic(self) -> float:
        """Return the controlled monotonic time."""
        return self.monotonic_time

    def sleep(self, seconds: float) -> None:
        """Record a sleep and advance both clocks without blocking."""
        self.advance(seconds)
        self.sleep_calls.append(seconds)

    def advance(self, seconds: float) -> None:
        """Advance both clocks by a non-negative duration."""
        if seconds < 0:
            raise ValueError("seconds must not be negative")
        self.current_time += timedelta(seconds=seconds)
        self.monotonic_time += seconds
