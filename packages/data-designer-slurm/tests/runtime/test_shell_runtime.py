# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("gpu_request_mode", ("gres", "visible"))
def test_bash_controller_scopes_secrets_cleans_steps_and_never_runs_host_python(
    tmp_path: Path,
    gpu_request_mode: str,
) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    attempt_directory = tmp_path / "runs/run-shell/shards/shard-00000/attempts/attempt-0001"
    log_directory = attempt_directory / "logs/execution-00000002"
    fake_bin = tmp_path / "bin"
    log_directory.mkdir(parents=True)
    fake_bin.mkdir()
    artifacts = tuple(_artifact(tmp_path, name) for name in ("runtime", "lock", "client", "server"))
    plan_path = tmp_path / "runs/run-shell/resolved-plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan = _plan(tmp_path, runtime_root, artifacts, gpu_request_mode)
    plan_path.write_text(json.dumps(plan))
    manifest_path = tmp_path / "manifest-source.json"
    manifest_path.write_text(json.dumps(_manifest(attempt_directory, artifacts, "a" * 64)))
    marker_path = tmp_path / "host-python-ran"
    _write_executable(fake_bin / "python3", f"#!/usr/bin/env bash\nprintf ran > {marker_path}\nexit 99\n")
    _write_executable(fake_bin / "curl", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "getent", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "scontrol", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "srun", _fake_srun())
    command = f"""
    set -Eeuo pipefail
DD_PLAN_SHA256={"a" * 64}
DD_SHARD_ID=shard-00000
DD_ATTEMPT_ORDINAL=0001
readonly DD_PLAN={shlex.quote(plan_path.as_posix())}
readonly DD_ATTEMPT_DIR={shlex.quote(attempt_directory.as_posix())}
source {shlex.quote((runtime_root / "entrypoint.sh").as_posix())}
dd_slurm_run_allocation "${{DD_PLAN}}" "${{DD_ATTEMPT_DIR}}"
"""
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_MANIFEST_SOURCE": manifest_path.as_posix(),
        "FAKE_GPU_MODE": gpu_request_mode,
        "SOURCE_TOKEN": "supersecret",
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_ARRAY_JOB_ID": "4101",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NODEID": "0",
    }

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker_path.exists()
    assert "supersecret" not in completed.stdout + completed.stderr + manifest_path.read_text()


def test_staged_shell_modules_parse_as_bash(tmp_path: Path) -> None:
    del tmp_path
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    scripts = tuple(runtime_root / name for name in ("entrypoint.sh", "plan_reader.sh", "step_runner.sh", "cleanup.sh"))

    completed = subprocess.run(("bash", "-n", *(path.as_posix() for path in scripts)), capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_shell_helpers_handle_empty_and_sparse_arrays() -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "plan_reader.sh").as_posix())}
source {shlex.quote((runtime_root / "cleanup.sh").as_posix())}
dd_read_null_values DD_VALUES < <(printf '')
(( ${{#DD_VALUES[@]}} == 0 ))
dd_start_runtime_timer
sleep 30 &
pid=$!
DD_MANAGED_PIDS[2]=${{pid}}
dd_cleanup_steps
! kill -0 "${{pid}}" 2>/dev/null
dd_stop_runtime_timer
"""

    completed = subprocess.run(("bash", "-c", command), capture_output=True, text=True, timeout=5)

    assert completed.returncode == 0, completed.stderr


def _artifact(root: Path, name: str) -> tuple[str, str]:
    path = root / f"{name}.artifact"
    content = name.encode()
    path.write_bytes(content)
    return path.as_posix(), hashlib.sha256(content).hexdigest()


def _plan(
    root: Path,
    runtime_root: Path,
    artifacts: tuple[tuple[str, str], ...],
    gpu_request_mode: str,
) -> dict[str, object]:
    runtime, lock, client, server = artifacts
    mounts = [
        {"source": root.as_posix(), "target": root.as_posix(), "read_only": False},
        {"source": runtime_root.as_posix(), "target": runtime_root.as_posix(), "read_only": False},
    ]
    return {
        "schema_version": 1,
        "runtime_bundle": {"path": runtime[0], "sha256": runtime[1]},
        "client": {
            "host_node_index": 0,
            "authored": {"cpus": 1},
            "dependency_lock": {"path": lock[0], "sha256": lock[1]},
            "image": {"path": client[0], "sha256": client[1]},
        },
        "resolved_gpus_per_node": 1,
        "selected_profile": {"profile": {"gpu_request_mode": gpu_request_mode}},
        "container_mounts": mounts,
        "deployments": [
            {
                "node_indices": [0],
                "image": {"path": server[0], "sha256": server[1]},
                "authored": {
                    "server": {"environment": {"SERVER_TOKEN": {"type": "secret", "environment": "SOURCE_TOKEN"}}}
                },
            }
        ],
        "builder": {"source": None},
        "shards": [{"array_task_index": 0, "input_partition": None}],
    }


def _manifest(
    attempt_directory: Path,
    artifacts: tuple[tuple[str, str], ...],
    plan_sha256: str,
) -> dict[str, object]:
    _, _, client, server = artifacts
    return {
        "schema_version": 1,
        "run_id": "run-shell",
        "shard_id": "shard-00000",
        "attempt_id": "attempt-0001",
        "plan_sha256": plan_sha256,
        "all_secret_environment_names": ["SERVER_TOKEN", "SOURCE_TOKEN"],
        "steps": [
            _step(attempt_directory, "client-preflight", "client_preflight", client[0], "fake-preflight"),
            _step(
                attempt_directory,
                "deployment-00000-replica-00000-rank-00000",
                "server",
                server[0],
                "fake-server",
                gpu_indices=[0],
                secret_environment={"SERVER_TOKEN": "SOURCE_TOKEN"},
                container_environment=["SERVER_TOKEN"],
                readiness={"host": "127.0.0.1", "port": 18000, "path": "/health", "deadline_seconds": 2},
            ),
            _step(
                attempt_directory,
                "deployment-00000-endpoint",
                "endpoint",
                client[0],
                "fake-endpoint",
                readiness={"host": "127.0.0.1", "port": 17000, "path": "/health", "deadline_seconds": 2},
            ),
            _step(
                attempt_directory,
                "client-generation",
                "client",
                client[0],
                "fake-client",
                secret_environment={"SOURCE_TOKEN": "SOURCE_TOKEN"},
                container_environment=["SOURCE_TOKEN"],
            ),
        ],
    }


def _step(
    attempt_directory: Path,
    step_id: str,
    role: str,
    image_path: str,
    command: str,
    *,
    gpu_indices: list[int] | None = None,
    secret_environment: dict[str, str] | None = None,
    container_environment: list[str] | None = None,
    readiness: dict[str, object] | None = None,
) -> dict[str, object]:
    log_root = attempt_directory / "logs/execution-00000002"
    return {
        "step_id": step_id,
        "role": role,
        "image_path": image_path,
        "command": [command],
        "cpus": 1,
        "gpu_indices": gpu_indices or [],
        "literal_environment": {"LC_ALL": "C"},
        "secret_environment": secret_environment or {},
        "environment_prefixes": {},
        "container_environment": container_environment or [],
        "stdout_path": (log_root / f"{step_id}.out").as_posix(),
        "stderr_path": (log_root / f"{step_id}.err").as_posix(),
        "launch_delay_seconds": 0,
        "readiness": readiness,
    }


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o700)


def _fake_srun() -> str:
    return """#!/usr/bin/env bash
set -Eeuo pipefail
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    [[ ${arguments[index]} == -- ]] && break
done
command=${arguments[index + 1]}
if [[ ${command} == python3 ]]; then
    operation=${arguments[index + 4]}
    [[ ! ${SOURCE_TOKEN+x} && ! ${SERVER_TOKEN+x} && ! ${CUDA_VISIBLE_DEVICES+x} ]]
    if [[ ${operation} == prepare ]]; then
        for ((position = index + 5; position < ${#arguments[@]}; position++)); do
            if [[ ${arguments[position]} == --manifest ]]; then
                cp "${FAKE_MANIFEST_SOURCE}" "${arguments[position + 1]}"
                break
            fi
        done
    fi
    exit 0
fi
case ${command} in
    fake-preflight)
        [[ ! ${SOURCE_TOKEN+x} && ! ${SERVER_TOKEN+x} && ! ${CUDA_VISIBLE_DEVICES+x} ]]
        ;;
    fake-client)
        [[ ${SOURCE_TOKEN} == supersecret && ! ${SERVER_TOKEN+x} && ! ${CUDA_VISIBLE_DEVICES+x} ]]
        ;;
    fake-server)
        [[ ${SERVER_TOKEN} == supersecret && ! ${SOURCE_TOKEN+x} ]]
        if [[ ${FAKE_GPU_MODE} == visible ]]; then
            [[ ${CUDA_VISIBLE_DEVICES} == 0 ]]
        else
            [[ ! ${CUDA_VISIBLE_DEVICES+x} ]]
        fi
        trap 'exit 0' TERM INT
        while :; do :; done
        ;;
    fake-endpoint)
        [[ ! ${SOURCE_TOKEN+x} && ! ${SERVER_TOKEN+x} && ! ${CUDA_VISIBLE_DEVICES+x} ]]
        trap 'exit 0' TERM INT
        while :; do :; done
        ;;
    *)
        exit 90
        ;;
esac
"""
