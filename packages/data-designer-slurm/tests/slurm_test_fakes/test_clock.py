# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest

from slurm_test_fakes import FakeClock


def test_fake_clock_advances_without_blocking(fake_clock: FakeClock) -> None:
    fake_clock.sleep(2.5)
    fake_clock.advance(1.5)

    assert fake_clock.now() == datetime(2026, 8, 18, 12, 0, 4, tzinfo=timezone.utc)
    assert fake_clock.monotonic() == 104.0
    assert fake_clock.sleep_calls == [2.5]


@pytest.mark.parametrize(
    "clock",
    (
        pytest.param(lambda: FakeClock(datetime(2026, 8, 18, 12)), id="naive-datetime"),
        pytest.param(
            lambda: FakeClock(datetime(2026, 8, 18, 13, tzinfo=timezone(timedelta(hours=1)))),
            id="non-utc-offset",
        ),
        pytest.param(
            lambda: FakeClock(datetime(2026, 8, 18, 12, tzinfo=timezone.utc), monotonic_time=-1),
            id="negative-monotonic",
        ),
    ),
)
def test_fake_clock_rejects_ambient_or_invalid_time(clock: Callable[[], FakeClock]) -> None:
    with pytest.raises(ValueError):
        clock()


def test_fake_clock_rejects_negative_advances(fake_clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        fake_clock.advance(-0.1)
