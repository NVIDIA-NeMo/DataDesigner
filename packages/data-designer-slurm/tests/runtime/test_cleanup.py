# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
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
    scheduler_bin = _write_fake_scheduler(tmp_path)
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
SLURM_JOB_ID=4101
DD_RUNTIME_MANIFEST=unused
DD_STEP_STDOUT={shlex.quote(str(tmp_path / "endpoint.out"))}
DD_STEP_STDERR={shlex.quote(str(tmp_path / "endpoint.err"))}
dd_read_step_ids() {{ DD_STEP_IDS=(endpoint); }}
dd_read_step() {{
    DD_STEP_EXECUTION=native
    DD_STEP_COMMAND=({shlex.quote(sys.executable)} {shlex.quote(str(worker))} endpoint {shlex.quote(str(events))} {tmp_path}/endpoint.ready "$backend")
}}
dd_build_srun_command() {{ DD_SRUN_COMMAND=(srun); }}
dd_materialize_step_environment() {{ :; }}
dd_start_endpoints
while [[ ! -f {tmp_path}/endpoint.ready ]]; do sleep 0.01; done
exit {exit_status}
"""
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "PATH": f"{scheduler_bin}:{os.environ['PATH']}"},
    )
    assert result.returncode == exit_status, result.stderr
    assert events.read_text().splitlines() == ["task_term", "endpoint", "backend"]
    assert (tmp_path / "endpoint.out.step-id").read_text() == "4101.7\n"


@pytest.mark.parametrize("identity", ("4101.7", "4102.7", "4101", "4101.batch", "4101.7 --all", None))
def test_endpoint_signal_is_bound_to_a_numeric_step_in_this_job(tmp_path: Path, identity: str | None) -> None:
    runtime = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    identity_file = tmp_path / "endpoint.step-id"
    if identity is not None:
        identity_file.write_text(identity + "\n")
    command = f"""
set -Eeuo pipefail
source {shlex.quote(str(runtime / "cleanup.sh"))}
SLURM_JOB_ID=4101
DD_ENDPOINT_PIDS=("$$")
DD_ENDPOINT_STEP_FILES[$$]={shlex.quote(str(identity_file))}
scancel() {{ printf '%s\\n' "$*" > {shlex.quote(str(tmp_path / "signal"))}; }}
dd_wait_steps() {{ :; }}
dd_stop_steps() {{ :; }}
dd_drain_endpoints
"""
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    signal = tmp_path / "signal"
    assert signal.exists() is (identity == "4101.7")
    if signal.exists():
        assert signal.read_text() == "--quiet --signal=TERM 4101.7\n"


def test_endpoint_cleanup_forces_launcher_only_after_drain_deadline(tmp_path: Path) -> None:
    runtime = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    command = f"""
set -Eeuo pipefail
source {shlex.quote(str(runtime / "cleanup.sh"))}
(trap '' TERM; touch {tmp_path}/ready; while :; do sleep 0.1; done) &
pid=$!
while [[ ! -f {tmp_path}/ready ]]; do sleep 0.01; done
SLURM_JOB_ID=4101
printf '4101.7\\n' > {tmp_path}/endpoint.step-id
DD_ENDPOINT_PIDS=("$pid")
DD_ENDPOINT_STEP_FILES[$pid]={tmp_path}/endpoint.step-id
scancel() {{ [[ $* == '--quiet --signal=TERM 4101.7' ]]; kill -TERM "$pid"; }}
dd_sleep() {{ kill -0 "$pid"; SECONDS=$((SECONDS + 1)); }}
started=$SECONDS
dd_drain_endpoints
((SECONDS - started >= 12))
! kill -0 "$pid" 2>/dev/null
"""
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr


def _write_fake_scheduler(directory: Path) -> Path:
    """Model srun's TERM-to-KILL forwarding separately from task signaling."""
    scheduler_bin = directory / "bin"
    scheduler_bin.mkdir()
    launcher = scheduler_bin / "srun"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, signal, subprocess, sys\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parents[1]\n"
        "child = subprocess.Popen(sys.argv[sys.argv.index('--') + 1:], "
        "env={**os.environ, 'SLURM_JOB_ID': '4101', 'SLURM_STEP_ID': '7'})\n"
        "def stop(*args):\n"
        "    with (root / 'events').open('a') as stream: stream.write('launcher_term\\n')\n"
        "    child.kill()\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "(root / 'task.pid').write_text(str(child.pid))\n"
        "status = child.wait()\n"
        "sys.exit(status if status >= 0 else 128 - status)\n"
    )
    cancel = scheduler_bin / "scancel"
    cancel.write_text(
        f"#!{sys.executable}\n"
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parents[1]\n"
        "assert sys.argv[1:] == ['--quiet', '--signal=TERM', '4101.7']\n"
        "with (root / 'events').open('a') as stream: stream.write('task_term\\n')\n"
        "os.kill(int((root / 'task.pid').read_text()), signal.SIGTERM)\n"
    )
    for path in (launcher, cancel):
        path.chmod(0o700)
    return scheduler_bin


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
