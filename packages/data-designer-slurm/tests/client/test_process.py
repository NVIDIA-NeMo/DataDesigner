# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.slurm.client.process import ClientWorkerProcess


def test_client_worker_process_uses_a_fresh_python_interpreter() -> None:
    commands: list[tuple[str, ...]] = []
    process = ClientWorkerProcess(
        executable="/client/python",
        executor=lambda command: commands.append(command) or 17,
    )

    assert process.run(("run", "--plan", "/workspace/resolved-plan.json")) == 17
    assert commands == [
        (
            "/client/python",
            "-m",
            "data_designer.slurm.client.worker",
            "run",
            "--plan",
            "/workspace/resolved-plan.json",
        )
    ]
