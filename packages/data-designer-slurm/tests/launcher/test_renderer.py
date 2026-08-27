# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import Literal

import pytest

from data_designer.slurm.config import SchedulerProfile, injected_profile
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.launcher.errors import SlurmBatchRenderError
from data_designer.slurm.launcher.renderer import render_generation_attempt_script
from data_designer.slurm.planning import ResolvedSlurmRunPlan, ResolvedSubmission

GOLDEN_DIRECTORY = Path(__file__).parents[1] / "slurm_test_fakes" / "golden" / "rendered"


@pytest.mark.parametrize(
    ("fixture_name", "plan_fixture"),
    (("single_node.sbatch", "single_node_plan"), ("multi_node.sbatch", "multi_node_plan")),
)
def test_renderer_matches_contract_bound_goldens(
    fixture_name: str,
    plan_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    plan = request.getfixturevalue(plan_fixture)

    assert render_generation_attempt_script(plan, attempt_ordinal=1) == (GOLDEN_DIRECTORY / fixture_name).read_text()


def test_renderer_omits_gres_for_visible_mode_and_emits_optional_submission_fields(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    profile = single_node_plan.selected_profile.profile.model_copy(
        update={
            "gpu_request_mode": "visible",
            "scheduler": SchedulerProfile(account="research", partition="batch"),
        }
    )
    plan = single_node_plan.model_copy(
        update={
            "selected_profile": injected_profile(profile),
            "submission": ResolvedSubmission(
                job_name="data-designer",
                account=None,
                partition=None,
                time_limit="01:00:00",
                comment="safe test run",
            ),
        }
    )

    script = render_generation_attempt_script(plan, attempt_ordinal=1)

    assert "#SBATCH --gres=" not in script
    assert "#SBATCH --account=" not in script
    assert "#SBATCH --partition=" not in script
    assert '#SBATCH --comment="safe test run"\n' in script


def test_renderer_emits_mem_per_gpu_for_gres_mode(single_node_plan: ResolvedSlurmRunPlan) -> None:
    profile = single_node_plan.selected_profile.profile.model_copy(
        update={"scheduler": SchedulerProfile(account="research", partition="batch", mem_per_gpu="80G")}
    )
    plan = single_node_plan.model_copy(update={"selected_profile": injected_profile(profile)})

    assert "#SBATCH --mem-per-gpu=80G\n" in render_generation_attempt_script(plan, attempt_ordinal=1)


@pytest.mark.parametrize("gpu_request_mode", ("gres", "visible"))
def test_renderer_reserves_client_cpus_for_each_gpu_request_mode(
    single_node_plan: ResolvedSlurmRunPlan,
    gpu_request_mode: Literal["gres", "visible"],
) -> None:
    profile = single_node_plan.selected_profile.profile.model_copy(update={"gpu_request_mode": gpu_request_mode})
    client = single_node_plan.client.model_copy(
        update={"authored": single_node_plan.client.authored.model_copy(update={"cpus": 17})}
    )
    plan = single_node_plan.model_copy(update={"selected_profile": injected_profile(profile), "client": client})

    assert "#SBATCH --cpus-per-task=17\n" in render_generation_attempt_script(plan, attempt_ordinal=1)


def test_renderer_rejects_mem_per_gpu_without_a_slurm_gpu_request(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    profile = single_node_plan.selected_profile.profile.model_copy(
        update={
            "gpu_request_mode": "visible",
            "scheduler": SchedulerProfile(account="research", partition="batch", mem_per_gpu="80G"),
        }
    )
    plan = single_node_plan.model_copy(update={"selected_profile": injected_profile(profile)})

    with pytest.raises(SlurmBatchRenderError, match="requires GRES"):
        render_generation_attempt_script(plan, attempt_ordinal=1)


def test_renderer_escapes_shell_expansion_in_structured_paths(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    plan = single_node_plan.model_copy(
        update={
            "runtime_bundle": ArtifactReference(
                path='/workspace/runtime/$(touch owned)-`whoami`-"bundle".tar.gz',
                sha256="e" * 64,
            )
        }
    )

    script = render_generation_attempt_script(plan, attempt_ordinal=1)

    assert (
        'readonly DD_RUNTIME_ARCHIVE="/workspace/runtime/\\$(touch owned)-\\`whoami\\`-\\"bundle\\".tar.gz"' in script
    )
    completed = subprocess.run(("bash", "-n"), input=script, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_renderer_keeps_user_text_on_one_non_executable_directive(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    comment = '$(touch owned)" --output=/tmp/owned; `whoami`'
    plan = single_node_plan.model_copy(
        update={"submission": single_node_plan.submission.model_copy(update={"comment": comment})}
    )

    script = render_generation_attempt_script(plan, attempt_ordinal=1)

    comment_lines = [line for line in script.splitlines() if line.startswith("#SBATCH --comment=")]
    assert len(comment_lines) == 1
    assert shlex.split(comment_lines[0].removeprefix("#SBATCH ")) == [f"--comment={comment}"]
    assert "\\$" not in comment_lines[0]
    assert "\\`" not in comment_lines[0]
    assert subprocess.run(("bash", "-n"), input=script, text=True, check=False).returncode == 0


@pytest.mark.parametrize(
    "attempt_ordinal",
    (0, -1, True),
)
def test_renderer_rejects_invalid_attempt_ordinals(
    single_node_plan: ResolvedSlurmRunPlan,
    attempt_ordinal: object,
) -> None:
    with pytest.raises(SlurmBatchRenderError, match="positive integer"):
        render_generation_attempt_script(single_node_plan, attempt_ordinal=attempt_ordinal)  # type: ignore[arg-type]


def test_renderer_rejects_control_characters_from_unvalidated_plan_copies(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    plan = single_node_plan.model_copy(
        update={"submission": single_node_plan.submission.model_copy(update={"comment": "unsafe\ntext"})}
    )

    with pytest.raises(SlurmBatchRenderError, match="control characters"):
        render_generation_attempt_script(plan, attempt_ordinal=1)


def test_renderer_is_a_thin_entrypoint(single_node_plan: ResolvedSlurmRunPlan) -> None:
    script = render_generation_attempt_script(single_node_plan, attempt_ordinal=12)

    assert script.count("dd_slurm_run_allocation") == 1
    assert 'readonly DD_ATTEMPT_ORDINAL="0012"' in script
    assert len(script.splitlines()) <= 42
    assert script.endswith("\n")


def test_rendered_script_verifies_exact_persisted_plan_bytes_before_sourcing_runtime(
    single_node_plan: ResolvedSlurmRunPlan,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    captured_plan_path = tmp_path / "captured-plan.json"
    entrypoint_path = tmp_path / "entrypoint.sh"
    entrypoint_path.write_text(
        f'dd_slurm_run_allocation() {{\n    cp -- "$1" {shlex.quote(str(captured_plan_path))}\n}}\n'
    )
    runtime_archive_path = tmp_path / "runtime.tar.gz"
    with tarfile.open(runtime_archive_path, mode="w:gz") as runtime_archive:
        runtime_archive.add(entrypoint_path, arcname="entrypoint.sh")

    plan = single_node_plan.model_copy(
        update={
            "authored_config": ArtifactReference(
                path=str(run_root / "authored-config.json"),
                sha256="a" * 64,
            ),
            "runtime_bundle": ArtifactReference(
                path=str(runtime_archive_path),
                sha256=hashlib.sha256(runtime_archive_path.read_bytes()).hexdigest(),
            ),
        }
    )
    plan_path = run_root / "resolved-plan.json"
    serialized_plan = plan.serialize_json()
    serialized_plan_bytes = serialized_plan.encode()
    plan_path.write_bytes(serialized_plan_bytes)
    script = render_generation_attempt_script(plan, attempt_ordinal=1)
    environment = {**os.environ, "SLURM_ARRAY_TASK_ID": "0"}

    valid = subprocess.run(
        ("bash",),
        input=script,
        env=environment,
        text=True,
        check=False,
    )

    assert valid.returncode == 0
    assert captured_plan_path.read_bytes() == serialized_plan_bytes

    captured_plan_path.unlink()
    plan_path.write_bytes(serialized_plan_bytes + b" ")
    tampered = subprocess.run(
        ("bash",),
        input=script,
        env=environment,
        text=True,
        check=False,
    )

    assert tampered.returncode != 0
    assert not captured_plan_path.exists()
