# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from conftest import RuntimeCase

from data_designer.slurm.config import ContainerMount
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.paths import get_container_path


def test_container_path_uses_most_specific_mount_and_preserves_relative_path(runtime_case: RuntimeCase) -> None:
    workspace = runtime_case.workspace.as_posix()
    nested = f"{workspace}/runs"
    plan = runtime_case.context.plan.model_copy(
        update={
            "container_mounts": (
                ContainerMount(source=workspace, target="/container/workspace"),
                ContainerMount(source=nested, target="/container/runs"),
            )
        }
    )

    mapped = get_container_path(plan, f"{nested}/run-single/resolved-plan.json")

    assert mapped == "/container/runs/run-single/resolved-plan.json"


def test_container_path_rejects_unmounted_and_read_only_writes(runtime_case: RuntimeCase) -> None:
    plan = runtime_case.context.plan.model_copy(
        update={
            "container_mounts": (
                ContainerMount(
                    source=runtime_case.workspace.as_posix(),
                    target="/container/workspace",
                    read_only=True,
                ),
            )
        }
    )

    with pytest.raises(SlurmRuntimeError, match="writable"):
        get_container_path(plan, runtime_case.context.attempt_directory.as_posix(), require_writable=True)
    with pytest.raises(SlurmRuntimeError, match="not available"):
        get_container_path(plan, "/different/root/file.json")
