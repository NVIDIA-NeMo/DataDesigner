# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_status", (0, 1, 130), ids=("completion", "failure", "cancellation"))
def test_allocation_keeps_backend_alive_until_endpoint_drains(tmp_path: Path, exit_status: int) -> None:
    runtime = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    events = tmp_path / "events"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "role, events, ready = sys.argv[1:4]\n"
        "def stop(*args):\n"
        "    if role == 'endpoint':\n"
        "        time.sleep(0.4)\n"
        "        os.kill(int(sys.argv[4]), 0)\n"
        "    with open(events, 'a') as stream: stream.write(role + '\\n')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "Path(ready).touch()\n"
        "while True: time.sleep(0.05)\n"
    )
    command = f"""
set -Eeuo pipefail
source {shlex.quote(str(runtime / "entrypoint.sh"))}
dd_cleanup_allocation_scratch() {{ :; }}
trap dd_runtime_exit EXIT
{shlex.quote(sys.executable)} {shlex.quote(str(worker))} backend {shlex.quote(str(events))} {tmp_path}/backend.ready &
backend=$!
DD_MANAGED_PIDS+=("$backend")
while [[ ! -f {tmp_path}/backend.ready ]]; do sleep 0.01; done
{shlex.quote(sys.executable)} {shlex.quote(str(worker))} endpoint {shlex.quote(str(events))} {tmp_path}/endpoint.ready "$backend" &
endpoint=$!
DD_MANAGED_PIDS+=("$endpoint")
DD_ENDPOINT_PIDS+=("$endpoint")
while [[ ! -f {tmp_path}/endpoint.ready ]]; do sleep 0.01; done
exit {exit_status}
"""
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=5)
    assert result.returncode == exit_status, result.stderr
    assert events.read_text().splitlines() == ["endpoint", "backend"]


def test_cleanup_kills_unresponsive_step_after_deadline(tmp_path: Path) -> None:
    runtime = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    command = f"""
set -Eeuo pipefail
source {shlex.quote(str(runtime / "cleanup.sh"))}
(trap '' TERM; touch {tmp_path}/ready; while :; do sleep 0.1; done) &
pid=$!
while [[ ! -f {tmp_path}/ready ]]; do sleep 0.01; done
dd_stop_steps 1 "$pid"
! kill -0 "$pid" 2>/dev/null
"""
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
