# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import data_designer
import data_designer.slurm

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_slurm_uses_shared_namespace() -> None:
    assert data_designer.__file__ is None
    assert data_designer.slurm.__name__ == "data_designer.slurm"


def test_slurm_is_published_before_base_extra() -> None:
    publish_script = (REPO_ROOT / "scripts" / "publish.sh").read_text()

    assert publish_script.index('"packages/data-designer-slurm"') < publish_script.index('"packages/data-designer"')


def test_slurm_registers_lazy_cli_extension() -> None:
    entry_points = importlib.metadata.entry_points(group="data_designer.cli")
    slurm_entry_point = next(
        entry_point
        for entry_point in entry_points
        if entry_point.name == "slurm" and entry_point.dist.name == "data-designer-slurm"
    )

    assert slurm_entry_point.value == "data_designer.slurm.cli:create_cli"
