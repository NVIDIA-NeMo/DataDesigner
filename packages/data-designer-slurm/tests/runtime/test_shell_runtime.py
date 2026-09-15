# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("gpu_request_mode", ("gres", "visible"))
def test_bash_controller_scopes_secrets_cleans_steps_and_never_runs_host_python(
    tmp_path: Path,
    gpu_request_mode: str,
) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    scratch_root = tmp_path / "data-designer-slurm-4101-0"
    runtime_copy = scratch_root / "runtime"
    shutil.copytree(runtime_root, runtime_copy)
    attempt_directory = tmp_path / "runs/run-shell/shards/shard-00000/attempts/attempt-0001"
    log_directory = attempt_directory / "logs/execution-00000002"
    fake_bin = tmp_path / "bin"
    log_directory.mkdir(parents=True)
    fake_bin.mkdir()
    artifacts = tuple(_artifact(tmp_path, name) for name in ("runtime", "lock", "client", "server"))
    plan_path = tmp_path / "runs/run-shell/resolved-plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan = _plan(tmp_path, runtime_copy, artifacts, gpu_request_mode)
    plan_path.write_text(json.dumps(plan))
    manifest_path = tmp_path / "manifest-source.json"
    manifest_path.write_text(json.dumps(_manifest(attempt_directory, artifacts, "a" * 64)))
    marker_path = tmp_path / "host-python-ran"
    _write_executable(fake_bin / "python3", f"#!/usr/bin/env bash\nprintf ran > {marker_path}\nexit 99\n")
    _write_executable(fake_bin / "curl", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "getent", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "scontrol", "#!/usr/bin/env bash\nprintf 'compute-001\\n'\n")
    _write_executable(fake_bin / "srun", _fake_srun())
    command = f"""
    set -Eeuo pipefail
DD_PLAN_SHA256={"a" * 64}
DD_SHARD_ID=shard-00000
DD_ATTEMPT_ORDINAL=0001
DD_RUNTIME_ARCHIVE={shlex.quote(artifacts[0][0])}
DD_RUNTIME_SHA256={artifacts[0][1]}
readonly DD_PLAN={shlex.quote(plan_path.as_posix())}
readonly DD_ATTEMPT_DIR={shlex.quote(attempt_directory.as_posix())}
source {shlex.quote((runtime_copy / "entrypoint.sh").as_posix())}
dd_slurm_run_allocation "${{DD_PLAN}}" "${{DD_ATTEMPT_DIR}}"
"""
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_MANIFEST_SOURCE": manifest_path.as_posix(),
        "FAKE_GPU_MODE": gpu_request_mode,
        "SOURCE_TOKEN": "supersecret",
        "CUDA_VISIBLE_DEVICES": "0",
        "DD_SCRATCH_ROOT": scratch_root.as_posix(),
        "SLURM_ARRAY_JOB_ID": "4101",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": "4101",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_JOB_NODELIST": "compute-001",
        "SLURM_JOB_GPUS": "0",
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
    assert not scratch_root.exists()
    assert not marker_path.exists()
    assert "supersecret" not in completed.stdout + completed.stderr + manifest_path.read_text()


def test_staged_shell_modules_parse_as_bash(tmp_path: Path) -> None:
    del tmp_path
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    scripts = tuple(
        runtime_root / name
        for name in ("scratch.sh", "entrypoint.sh", "plan_reader.sh", "step_runner.sh", "cleanup.sh")
    )

    completed = subprocess.run(("bash", "-n", *(path.as_posix() for path in scripts)), capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_scratch_prefers_slurm_tmpdir_and_cleans_private_tree(tmp_path: Path) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    slurm_tmp = tmp_path / "slurm-tmp"
    fallback_tmp = tmp_path / "fallback-tmp"
    slurm_tmp.mkdir()
    fallback_tmp.mkdir()
    expected_root = slurm_tmp / "data-designer-slurm-4101-7"
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "scratch.sh").as_posix())}
dd_initialize_local_scratch
[[ $DD_SCRATCH_ROOT == {shlex.quote(expected_root.as_posix())} ]]
[[ $HOME == "${{DD_SCRATCH_ROOT}}/home" ]]
[[ $ENROOT_TEMP_PATH == "${{DD_SCRATCH_ROOT}}/enroot/tmp" ]]
dd_remove_local_scratch "${{DD_SCRATCH_ROOT}}"
[[ ! -e $DD_SCRATCH_ROOT ]]
"""

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_JOB_ID": "4101",
            "SLURM_TMPDIR": slurm_tmp.as_posix(),
            "TMPDIR": fallback_tmp.as_posix(),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert not expected_root.exists()


def test_scratch_falls_back_to_tmpdir_and_rejects_symlink_root(tmp_path: Path) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    fallback_tmp = tmp_path / "fallback-tmp"
    outside = tmp_path / "outside"
    fallback_tmp.mkdir()
    outside.mkdir()
    expected_root = fallback_tmp / "data-designer-slurm-4101-0"
    expected_root.symlink_to(outside, target_is_directory=True)
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "scratch.sh").as_posix())}
if dd_initialize_local_scratch; then
    exit 1
else
    [[ $? == 73 ]]
fi
"""

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SLURM_JOB_ID": "4101",
            "SLURM_TMPDIR": (tmp_path / "missing").as_posix(),
            "TMPDIR": fallback_tmp.as_posix(),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_root.is_symlink()
    assert not tuple(outside.iterdir())


def test_scratch_rejects_inconsistent_allocation_root(tmp_path: Path) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    expected_root = scratch_parent / "data-designer-slurm-4101-0"
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "scratch.sh").as_posix())}
if dd_initialize_local_scratch /different/data-designer-slurm-4101-0; then
    exit 1
else
    [[ $? == 73 ]]
fi
"""

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        env={**os.environ, "SLURM_JOB_ID": "4101", "SLURM_TMPDIR": scratch_parent.as_posix()},
    )

    assert completed.returncode == 0, completed.stderr
    assert not expected_root.exists()


def test_runtime_bundle_rejects_digest_mismatch_and_existing_target(tmp_path: Path) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        bundle.add(runtime_root / "scratch.sh", arcname="scratch.sh")
        bundle.add(runtime_root / "entrypoint.sh", arcname="entrypoint.sh")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "scratch.sh").as_posix())}
dd_initialize_local_scratch
if dd_extract_runtime_bundle {shlex.quote(archive.as_posix())} {"0" * 64}; then
    exit 1
else
    [[ $? == 65 ]]
fi
mkdir "${{DD_SCRATCH_ROOT}}/runtime"
if dd_extract_runtime_bundle {shlex.quote(archive.as_posix())} {digest}; then
    exit 1
else
    [[ $? == 73 ]]
fi
dd_remove_local_scratch "${{DD_SCRATCH_ROOT}}"
"""

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        env={**os.environ, "SLURM_JOB_ID": "4101", "SLURM_TMPDIR": scratch_parent.as_posix()},
    )

    assert completed.returncode == 0, completed.stderr


def test_scratch_stages_runtime_for_fake_multi_node_allocation(tmp_path: Path) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    scratch_parent = tmp_path / "scratch"
    fake_bin = tmp_path / "bin"
    scratch_parent.mkdir()
    fake_bin.mkdir()
    entrypoint = tmp_path / "entrypoint.sh"
    entrypoint.write_text("true\n")
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        bundle.add(runtime_root / "scratch.sh", arcname="scratch.sh")
        bundle.add(entrypoint, arcname="entrypoint.sh")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    srun_log = tmp_path / "srun.log"
    _write_executable(
        fake_bin / "srun",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${FAKE_SRUN_LOG}"\n'
        "while [[ $# -gt 0 && $1 != -- ]]; do shift; done\n"
        "shift\n"
        'exec "$@"\n',
    )
    expected_root = scratch_parent / "data-designer-slurm-4101-0"
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "scratch.sh").as_posix())}
DD_RUNTIME_ARCHIVE={shlex.quote(archive.as_posix())}
DD_RUNTIME_SHA256={digest}
dd_initialize_local_scratch
dd_stage_allocation_runtime
[[ -f "${{DD_SCRATCH_ROOT}}/runtime/entrypoint.sh" ]]
dd_cleanup_allocation_scratch
"""

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FAKE_SRUN_LOG": srun_log.as_posix(),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "4101",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_TMPDIR": scratch_parent.as_posix(),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "--nodes=2 --ntasks=2" in srun_log.read_text()
    assert "--time=00:01:00 --wait=0" in srun_log.read_text()
    assert not expected_root.exists()


@pytest.mark.parametrize(
    ("visible_gpus", "expected_gpus", "expected_status"),
    (("0,1,2,3,4,5,6,7", 1, 0), ("0", 2, 65)),
)
def test_bash_gpu_count_requires_planned_minimum(
    visible_gpus: str,
    expected_gpus: int,
    expected_status: int,
) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "entrypoint.sh").as_posix())}
DD_EXPECTED_GPUS={expected_gpus}
dd_verify_gpu_count
"""

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_gpus},
    )

    assert completed.returncode == expected_status


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


def test_step_runner_builds_one_coordinated_srun_across_selected_nodes() -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "step_runner.sh").as_posix())}
DD_STEP_NODE_HOSTS=(compute-001 compute-002)
DD_STEP_GPU_INDICES=(0 1)
DD_STEP_CONTAINER_ENV=()
DD_STEP_CPUS=4
DD_STEP_IMAGE=/images/server.sqsh
DD_STEP_KILL_ON_BAD_EXIT=true
DD_GPU_REQUEST_MODE=gres
DD_CONTAINER_MOUNTS=
DD_SCRATCH_ROOT=/tmp/data-designer-slurm-4101-0
DD_SCRATCH_CONTAINER_ROOT=/run/data-designer-slurm
dd_build_srun_command
command=${{DD_SRUN_COMMAND[*]}}
[[ $command == *--nodes=2* ]]
[[ $command == *--ntasks=2* ]]
[[ $command == *--ntasks-per-node=1* ]]
[[ $command == *--nodelist=compute-001,compute-002* ]]
[[ $command == *--kill-on-bad-exit=1* ]]
[[ $command == *--gpus-per-task=2* ]]
[[ $command == *--container-mounts=/tmp/data-designer-slurm-4101-0:/run/data-designer-slurm* ]]
"""

    completed = subprocess.run(("bash", "-c", command), capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_shell_resolves_scheduler_hosts_in_planner_order(tmp_path: Path) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "scontrol", "#!/usr/bin/env bash\nprintf 'compute-001\\ncompute-002\\n'\n")
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "entrypoint.sh").as_posix())}
DD_EXPECTED_NODES=2
DD_CLIENT_NODE_INDEX=0
SLURM_JOB_NODELIST=compute-[001-002]
dd_resolve_allocation_hosts
[[ ${{DD_ALLOCATION_HOSTS[*]}} == 'compute-001 compute-002' ]]
[[ $DD_CLIENT_HOST == compute-001 ]]
DD_EXPECTED_NODES=3
! dd_resolve_allocation_hosts
"""

    completed = subprocess.run(
        ("bash", "-c", command),
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 0, completed.stderr


def test_plan_reader_rejects_mount_overlapping_allocation_scratch(tmp_path: Path) -> None:
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    artifacts = tuple(_artifact(tmp_path, name) for name in ("runtime", "lock", "client", "server"))
    payload = _plan(tmp_path, tmp_path / "runtime-root", artifacts, "gres")
    mounts = payload["container_mounts"]
    assert isinstance(mounts, list)
    mounts.append({"source": "/source", "target": "/run", "read_only": False})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))
    command = f"""
set -Eeuo pipefail
source {shlex.quote((runtime_root / "plan_reader.sh").as_posix())}
if dd_read_control_plan {shlex.quote(plan_path.as_posix())}; then
    exit 1
fi
"""

    completed = subprocess.run(("bash", "-c", command), capture_output=True, text=True)

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
        "node_hosts": ["compute-001"],
        "kill_on_bad_exit": False,
        "literal_environment": {"LC_ALL": "C"},
        "secret_environment": secret_environment or {},
        "environment_prefixes": {},
        "container_environment": container_environment or [],
        "stdout_path": (log_root / f"{step_id}.out").as_posix(),
        "stderr_path": (log_root / f"{step_id}.err").as_posix(),
        "launch_delay_seconds": 0,
        "readiness": [readiness] if readiness is not None else [],
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
    [[ ${arguments[*]} == *--container-env=PYTHONPATH,SLURM_JOB_GPUS* ]]
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
