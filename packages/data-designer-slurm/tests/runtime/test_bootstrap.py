# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from conftest import RuntimeCase

from data_designer.slurm.runtime.bootstrap import RuntimeBootstrapManifest, build_runtime_manifest
from data_designer.slurm.runtime.models import RuntimeStepRole


def test_bootstrap_manifest_builds_typed_one_node_steps_without_secret_values(runtime_case: RuntimeCase) -> None:
    context = runtime_case.context
    runtime_root = context.attempt_directory / "runtime"
    log_directory = context.attempt_directory / "logs/execution-00000002"

    manifest = build_runtime_manifest(
        context,
        runtime_root=runtime_root,
        log_directory=log_directory,
    )
    reloaded = RuntimeBootstrapManifest.model_validate_json(manifest.serialize_json())

    assert reloaded == manifest
    assert [step.role for step in manifest.steps] == [
        RuntimeStepRole.CLIENT_PREFLIGHT,
        RuntimeStepRole.SERVER,
        RuntimeStepRole.ENDPOINT,
        RuntimeStepRole.CLIENT,
    ]
    assert all(step.command[0] != "srun" for step in manifest.steps)
    assert all(step.stdout_path.startswith(context.attempt_directory.as_posix()) for step in manifest.steps)
    assert manifest.steps[-1].command[:4] == (
        "python3",
        "-m",
        "data_designer.slurm.runtime.entrypoint",
        "client",
    )
