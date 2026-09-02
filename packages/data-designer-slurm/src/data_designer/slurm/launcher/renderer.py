# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal safe rendering for thin deterministic Slurm batch entrypoints."""

from __future__ import annotations

import posixpath

from data_designer.slurm.launcher.batch import quote_shell_value, render_batch_directives
from data_designer.slurm.launcher.errors import SlurmBatchRenderError
from data_designer.slurm.planning import ResolvedSlurmRunPlan


def render_generation_attempt_script(plan: ResolvedSlurmRunPlan, *, attempt_ordinal: int) -> str:
    """Render a resolved generation plan as one thin deterministic entrypoint."""
    if type(attempt_ordinal) is not int or attempt_ordinal <= 0:
        raise SlurmBatchRenderError("attempt_ordinal must be a positive integer")

    run_root = posixpath.dirname(plan.authored_config.path)
    plan_path = posixpath.join(run_root, "resolved-plan.json")
    directive_text = render_batch_directives(_build_generation_directives(plan))
    attempt = f"{attempt_ordinal:04d}"

    return f"""#!/usr/bin/env bash
{directive_text}
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

readonly DD_RUNTIME_ARCHIVE={quote_shell_value(plan.runtime_bundle.path)}
readonly DD_RUNTIME_SHA256={quote_shell_value(plan.runtime_bundle.sha256)}
readonly DD_PLAN={quote_shell_value(plan_path)}
readonly DD_PLAN_SHA256={quote_shell_value(plan.compute_sha256())}
readonly DD_RUN_ROOT={quote_shell_value(run_root)}
readonly DD_ATTEMPT_ORDINAL={quote_shell_value(attempt)}

verify_sha256() {{
    local actual_sha256
    actual_sha256="$(sha256sum < "$2")"
    [[ "${{actual_sha256%% *}}" == "$1" ]]
}}

verify_sha256 "${{DD_RUNTIME_SHA256}}" "${{DD_RUNTIME_ARCHIVE}}"
verify_sha256 "${{DD_PLAN_SHA256}}" "${{DD_PLAN}}"
if [[ ! ${{SLURM_ARRAY_TASK_ID:-}} =~ ^[0-9]+$ ]]; then
    printf '%s\\n' 'SLURM_ARRAY_TASK_ID must be a non-negative integer' >&2
    exit 64
fi
readonly DD_ARRAY_TASK_ID="${{SLURM_ARRAY_TASK_ID}}"
printf -v DD_SHARD_ID 'shard-%05d' "${{DD_ARRAY_TASK_ID}}"
readonly DD_SHARD_ID
readonly DD_ATTEMPT_DIR="${{DD_RUN_ROOT}}/shards/${{DD_SHARD_ID}}/attempts/attempt-${{DD_ATTEMPT_ORDINAL}}"
install -d -m 0700 "${{DD_ATTEMPT_DIR}}"
DD_RUNTIME_DIR="$(mktemp -d "${{DD_ATTEMPT_DIR}}/runtime.${{DD_RUNTIME_SHA256}}.XXXXXX")"
readonly DD_RUNTIME_DIR
tar -xzf "${{DD_RUNTIME_ARCHIVE}}" -C "${{DD_RUNTIME_DIR}}"

source "${{DD_RUNTIME_DIR}}/entrypoint.sh"
dd_slurm_run_allocation "${{DD_PLAN}}" "${{DD_ATTEMPT_DIR}}"
"""


def _build_generation_directives(plan: ResolvedSlurmRunPlan) -> tuple[tuple[str, str | None], ...]:
    node_indices = (
        plan.client.host_node_index,
        *(index for deployment in plan.deployments for index in deployment.node_indices),
    )
    node_count = max(node_indices) + 1
    array = "0"
    if plan.array_tasks.count > 1:
        array = f"0-{plan.array_tasks.count - 1}"
        if plan.array_tasks.max_concurrent is not None:
            array = f"{array}%{plan.array_tasks.max_concurrent}"

    values: list[tuple[str, str | None]] = [
        ("job-name", plan.submission.job_name),
        ("account", plan.submission.account),
        ("partition", plan.submission.partition),
        ("nodes", str(node_count)),
        ("cpus-per-task", str(plan.client.authored.cpus)),
        ("time", plan.submission.time_limit),
        ("array", array),
    ]
    profile = plan.selected_profile.profile
    if profile.gpu_request_mode == "gres":
        values.append(("gres", f"gpu:{plan.resolved_gpus_per_node}"))
    elif profile.scheduler.mem_per_gpu is not None:
        raise SlurmBatchRenderError("mem_per_gpu requires GRES GPU request mode")
    if profile.scheduler.mem_per_gpu is not None:
        values.append(("mem-per-gpu", profile.scheduler.mem_per_gpu))
    if plan.submission.comment is not None:
        values.append(("comment", plan.submission.comment))
    return tuple(values)
