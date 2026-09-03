# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.slurm.runtime.entrypoint import main


def test_entrypoint_rejects_relative_paths_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(("--plan", "resolved-plan.json", "--attempt-dir", "attempt-0001")) == 64
    captured = capsys.readouterr()
    assert "runtime paths must be absolute" in captured.err
    assert "Traceback" not in captured.err
