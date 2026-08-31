# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import posixpath
from pathlib import Path

from data_designer.slurm.planning import ResolvedSlurmRunPlan

GOLDEN_DIRECTORY = Path(__file__).parent / "golden" / "rendered"


def test_rendered_script_fixtures_are_pinned_and_bound_to_canonical_plans(
    single_node_plan: ResolvedSlurmRunPlan,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _assert_script_matches_plan(
        single_node_plan,
        "single_node.sbatch",
        expected_fixture_sha256="65032718bfc8fbd6c60700add08ebc098ba2e64858c23808decbefe5a8153d91",
    )
    _assert_script_matches_plan(
        multi_node_plan,
        "multi_node.sbatch",
        expected_fixture_sha256="58c3a2eddc9a0d377eb0a90655e70a6b2484cade0c52b2a01b9b388dcd4667e2",
    )


def test_rendered_scripts_contain_no_site_or_user_data() -> None:
    contents = "\n".join(path.read_text().casefold() for path in GOLDEN_DIRECTORY.glob("*.sbatch"))

    for forbidden in (
        "nvidia",
        "cluster",
        "login",
        "lustre",
        "/home/",
        "/users/",
    ):
        assert forbidden not in contents


def _assert_script_matches_plan(
    plan: ResolvedSlurmRunPlan,
    filename: str,
    *,
    expected_fixture_sha256: str,
) -> None:
    script = (GOLDEN_DIRECTORY / filename).read_text()
    node_indices = (
        plan.client.host_node_index,
        *(index for deployment in plan.deployments for index in deployment.node_indices),
    )
    node_count = max(node_indices) + 1
    array = "0" if plan.array_tasks.count == 1 else f"0-{plan.array_tasks.count - 1}"
    if plan.array_tasks.count > 1 and plan.array_tasks.max_concurrent is not None:
        array = f"{array}%{plan.array_tasks.max_concurrent}"
    plan_path = posixpath.join(posixpath.dirname(plan.authored_config.path), "resolved-plan.json")
    run_root = posixpath.dirname(plan.authored_config.path)

    assert f"#SBATCH --job-name={plan.submission.job_name}\n" in script
    assert f"#SBATCH --account={plan.submission.account}\n" in script
    assert f"#SBATCH --partition={plan.submission.partition}\n" in script
    assert f"#SBATCH --nodes={node_count}\n" in script
    assert f"#SBATCH --cpus-per-task={plan.client.authored.cpus}\n" in script
    assert f"#SBATCH --time={plan.submission.time_limit}\n" in script
    assert f"#SBATCH --array={array}\n" in script
    assert f"#SBATCH --gres=gpu:{plan.resolved_gpus_per_node}\n" in script
    assert f'readonly DD_RUNTIME_ARCHIVE="{plan.runtime_bundle.path}"\n' in script
    assert f'readonly DD_RUNTIME_SHA256="{plan.runtime_bundle.sha256}"\n' in script
    assert f'readonly DD_PLAN="{plan_path}"\n' in script
    assert f'readonly DD_PLAN_SHA256="{plan.compute_sha256()}"\n' in script
    assert f'readonly DD_RUN_ROOT="{run_root}"\n' in script
    assert script.count("dd_slurm_run_allocation") == 1
    assert script.endswith("\n")
    assert hashlib.sha256(script.encode()).hexdigest() == expected_fixture_sha256
