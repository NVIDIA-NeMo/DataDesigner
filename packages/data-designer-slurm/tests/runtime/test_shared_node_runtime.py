# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import RuntimeCase, relocate_plan

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.bootstrap import build_runtime_manifest
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.models import RuntimeStepRole
from data_designer.slurm.runtime.node_spec import (
    NodeProcessSpec,
    NodeSpec,
    NodeWorkerSpec,
    decode_node_worker_spec,
    encode_node_worker_spec,
)
from data_designer.slurm.runtime.node_worker import NodeProcessSupervisor, main
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.runtime.steps import build_vllm_steps
from data_designer.slurm.serving.resolver import resolve_vllm_server
from data_designer.slurm.state import RetryPlan, RetryShard


def test_shared_worker_has_both_models_readiness_and_gpu_free_clients(
    runtime_case: RuntimeCase, shared_node_plan: ResolvedSlurmRunPlan, tmp_path: Path
) -> None:
    plan = relocate_plan(shared_node_plan, runtime_case.workspace)
    context = replace(runtime_case.context, plan=plan, shard=plan.shards[0])
    manifest = build_runtime_manifest(
        context,
        {"SLURM_JOB_GPUS": "1,3,5,7,9,11,13,15"},
        runtime_root=Path("/tmp/data-designer-slurm-4101-0/runtime"),
        log_directory=context.attempt_directory / "logs",
        layout=AllocationLayout(("compute-001",)),
    )
    servers = [step for step in manifest.steps if step.role is RuntimeStepRole.SERVER]
    assert len(servers) == 1
    server = servers[0]
    assert server.gpu_indices == tuple(range(8))
    assert server.gpu_binding == "none"
    assert len(server.readiness) == 4
    assert len({probe.port for probe in server.readiness}) == 4
    worker = decode_node_worker_spec(server.command[-1])
    processes = worker.nodes[0].processes
    assert [process.gpu_indices for process in processes] == [(0, 1), (4, 5), (2, 3), (6, 7)]
    assert [process.launch_delay_seconds for process in processes] == [0, 0, 65, 65]
    assert {process.command[2] for process in processes} == {
        deployment.authored.model for deployment in plan.deployments
    }
    endpoints = [step for step in manifest.steps if step.role is RuntimeStepRole.ENDPOINT]
    assert len(endpoints) == 2
    assert all(not step.gpu_indices for step in manifest.steps if step.execution == "native")
    assert all(len(step.readiness) == 1 for step in endpoints)
    client = manifest.steps[-1]
    assert client.command.count("--endpoint") == 2
    assert any(argument.startswith("writer=http://") for argument in client.command)
    assert any(argument.startswith("judge=http://") for argument in client.command)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.serialize_json())
    runtime_root = Path(__file__).parents[2] / "src/data_designer/slurm/runtime"
    script = f"""
set -Eeuo pipefail
source {shlex.quote(str(runtime_root / "plan_reader.sh"))}
source {shlex.quote(str(runtime_root / "step_runner.sh"))}
DD_GPU_REQUEST_MODE=gres
DD_CONTAINER_MOUNTS=
DD_SCRATCH_ROOT=/tmp/data-designer-slurm-4101-0
DD_SCRATCH_CONTAINER_ROOT=/run/data-designer-slurm
dd_read_step {shlex.quote(str(manifest_path))} {shlex.quote(server.step_id)}
dd_build_srun_command
printf '%s\n' "${{DD_SRUN_COMMAND[@]}}"
"""
    result = subprocess.run(("bash", "-c", script), capture_output=True, text=True, check=True)
    arguments = result.stdout.splitlines()
    assert "--nodes=1" in arguments
    assert "--gpus-per-task=8" in arguments
    assert "--gpu-bind=none" in arguments
    assert "--overlap" in arguments
    assert not any(argument.startswith("--exclusive") for argument in arguments)


@pytest.mark.parametrize("visible", (("1", "3", "5", "7"), ("0", "1", "2", "3"), ("GPU-a", "GPU-b", "GPU-c", "GPU-d")))
def test_distinct_model_children_and_their_children_receive_disjoint_devices(
    tmp_path: Path, visible: tuple[str, ...]
) -> None:
    files = (tmp_path / "writer", tmp_path / "judge")
    programs = []
    for path in files:
        child = "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"
        program = (
            "import os,signal,subprocess,sys,time; from pathlib import Path; "
            f"value=subprocess.check_output([sys.executable,'-c',{child!r}],text=True); "
            f"Path({str(path)!r}).write_text(os.environ['CUDA_VISIBLE_DEVICES']+'\\n'+value); "
            "time.sleep(0.4); sys.exit(23)"
        )
        programs.append((sys.executable, "-c", program))
    node = NodeSpec(
        0,
        "compute-001",
        (),
        (
            NodeProcessSpec("writer", programs[0], (0, 1), 0),
            NodeProcessSpec("judge", programs[1], (2, 3), 0),
        ),
    )
    supervisor = NodeProcessSupervisor(environment={"PATH": os.defpath})
    assert supervisor.run(node, visible) == 23
    assert supervisor.cleanup_complete
    assert files[0].read_text().splitlines() == [",".join(visible[:2])] * 2
    assert files[1].read_text().splitlines() == [",".join(visible[2:])] * 2


def test_shared_preflight_verifies_every_model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    node = NodeSpec(
        0,
        "compute-001",
        (),
        (NodeProcessSpec("writer", ("true",), (0,), 0), NodeProcessSpec("judge", ("true",), (1,), 0)),
    )
    spec = NodeWorkerSpec(1, 2, None, (node,), (str(first), str(second)))
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURMD_NODENAME", "compute-001")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert decode_node_worker_spec(encode_node_worker_spec(spec)) == spec
    assert main(("preflight", "--spec", encode_node_worker_spec(spec))) == 70
    second.mkdir()
    assert main(("preflight", "--spec", encode_node_worker_spec(spec))) == 0


def test_model_failure_interrupts_sibling_launch_standoff() -> None:
    node = NodeSpec(
        0,
        "compute-001",
        (),
        (
            NodeProcessSpec("writer", (sys.executable, "-c", "raise SystemExit(23)"), (0,), 0),
            NodeProcessSpec("judge", ("must-not-start",), (1,), 30),
        ),
    )
    supervisor = NodeProcessSupervisor(environment={})
    started = time.monotonic()
    assert supervisor.run(node, ("0", "1")) == 23
    assert time.monotonic() - started < 2
    assert supervisor.cleanup_complete


def test_independent_steps_reject_shared_plan_at_entry(
    shared_node_plan: ResolvedSlurmRunPlan,
    tmp_path: Path,
) -> None:
    server = resolve_vllm_server(shared_node_plan, shared_node_plan.deployments[0].deployment_id)
    with pytest.raises(SlurmRuntimeError, match="coordinated"):
        build_vllm_steps(server, shared_node_plan, tmp_path, {}, tmp_path)


def test_fresh_process_retry_preserves_shared_placement(
    shared_node_plan: ResolvedSlurmRunPlan,
    runtime_case: RuntimeCase,
    tmp_path: Path,
) -> None:
    plan = shared_node_plan
    retry = RetryPlan(
        schema_version=1,
        retry_id="retry-0001",
        run_id=plan.run_id,
        created_at=runtime_case.created_at,
        resolved_plan=ArtifactReference(
            path=str(Path(plan.authored_config.path).with_name("resolved-plan.json")), sha256=plan.compute_sha256()
        ),
        planned_shards=(
            RetryShard(
                shard_id=plan.shards[0].shard_id, attempt_id="attempt-0002", attempt_ordinal=2, array_task_index=0
            ),
        ),
        effective_resume_mode="never",
    )
    plan_path, retry_path = tmp_path / "plan.json", tmp_path / "retry.json"
    plan_path.write_text(plan.serialize_json())
    retry_path.write_text(retry.serialize_json())
    program = (
        "import json,sys; from pathlib import Path; "
        "from data_designer.slurm.planning import ResolvedSlurmRunPlan; "
        "from data_designer.slurm.state import RetryPlan; "
        "from data_designer.slurm.launcher.renderer import render_generation_retry_script; "
        "from data_designer.slurm.serving.resolver import resolve_vllm_server; "
        "plan=ResolvedSlurmRunPlan.model_validate_json(Path(sys.argv[1]).read_text()); "
        "retry=RetryPlan.model_validate_json(Path(sys.argv[2]).read_text()); "
        "print(json.dumps({'script':render_generation_retry_script(plan,retry),"
        "'gpus':[list(p.gpu_indices) for d in plan.deployments for p in resolve_vllm_server(plan,d.deployment_id).processes]}))"
    )
    result = subprocess.run(
        (sys.executable, "-c", program, str(plan_path), str(retry_path)), capture_output=True, text=True, check=True
    )
    evidence = json.loads(result.stdout)
    assert evidence["gpus"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert "#SBATCH --nodes=1\n" in evidence["script"]
    assert "#SBATCH --gres=gpu:8\n" in evidence["script"]
    assert "#SBATCH --exclusive" not in evidence["script"]
