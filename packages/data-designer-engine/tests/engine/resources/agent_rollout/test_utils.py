# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from data_designer.engine.resources.agent_rollout.utils import min_max_timestamps, parse_iso8601


@pytest.mark.parametrize(
    ("timestamps", "expected"),
    [
        pytest.param([], (None, None), id="empty"),
        pytest.param(
            ["2025-01-01T00:30:00+01:00", "2025-01-01T00:00:00Z"],
            ("2025-01-01T00:30:00+01:00", "2025-01-01T00:00:00Z"),
            id="mixed-offset-lex-disagrees-with-chrono",
        ),
        pytest.param(
            ["2025-01-01T00:00:00.500Z", "2025-01-01T00:00:00Z"],
            ("2025-01-01T00:00:00Z", "2025-01-01T00:00:00.500Z"),
            id="mixed-precision",
        ),
        pytest.param(
            ["2025-01-01T00:00:00", "2025-01-02T00:00:00Z"],
            ("2025-01-01T00:00:00", "2025-01-02T00:00:00Z"),
            id="naive-treated-as-utc-and-compared-against-aware",
        ),
        pytest.param(
            ["not-a-timestamp", "2025-01-01T00:00:00Z"],
            ("2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
            id="unparseable-values-skipped",
        ),
        pytest.param(["not-a-timestamp"], (None, None), id="only-unparseable"),
    ],
)
def test_min_max_timestamps(timestamps: list[str], expected: tuple[str | None, str | None]) -> None:
    assert min_max_timestamps(timestamps) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("2025-01-01T00:00:00Z", datetime(2025, 1, 1, tzinfo=UTC), id="utc-z"),
        pytest.param(
            "2025-01-01T00:00:00.123Z",
            datetime(2025, 1, 1, microsecond=123000, tzinfo=UTC),
            id="fractional-utc-z",
        ),
        pytest.param("2025-01-01T00:00:00+00:00", datetime(2025, 1, 1, tzinfo=UTC), id="utc-offset"),
        pytest.param(
            "2025-01-01T00:00:00+01:00",
            datetime(2025, 1, 1, tzinfo=timezone(timedelta(hours=1))),
            id="non-utc-offset",
        ),
        pytest.param("2025-01-01T00:00:00", datetime(2025, 1, 1, tzinfo=UTC), id="naive-as-utc"),
        pytest.param("not-a-timestamp", None, id="malformed"),
        pytest.param("", None, id="empty"),
    ],
)
def test_parse_iso8601(value: str, expected: datetime | None) -> None:
    result = parse_iso8601(value)

    assert result == expected
    if expected is not None:
        assert result is not None
        assert result.utcoffset() == expected.utcoffset()
