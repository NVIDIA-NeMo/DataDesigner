# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest

import data_designer.engine.dataset_builders.utils.async_concurrency as async_concurrency


def test_run_async_scheduler_starts_loop_before_creating_coroutine(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = Mock()
    monkeypatch.setattr(
        async_concurrency,
        "ensure_async_engine_loop",
        Mock(side_effect=RuntimeError("loop startup failed")),
    )

    with pytest.raises(RuntimeError, match="loop startup failed"):
        async_concurrency.run_async_scheduler(scheduler)

    scheduler.run.assert_not_called()
