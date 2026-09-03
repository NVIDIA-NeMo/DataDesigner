# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal safe rendering for thin deterministic Slurm batch entrypoints."""

from __future__ import annotations

import posixpath

from data_designer.slurm.images.records import validate_enroot_mount_path
from data_designer.slurm.launcher.batch import quote_shell_value, render_batch_directives
from data_designer.slurm.launcher.errors import SlurmBatchRenderError
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.outputs import RetryPlan


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


def render_generation_retry_script(plan: ResolvedSlurmRunPlan, retry: RetryPlan) -> str:
    """Render selected failed shards as one immutable retry array submission."""
    run_root = posixpath.dirname(plan.authored_config.path)
    plan_path = posixpath.join(run_root, "resolved-plan.json")
    if retry.run_id != plan.run_id:
        raise SlurmBatchRenderError("retry run identity does not match the resolved plan")
    if retry.resolved_plan.path != plan_path or retry.resolved_plan.sha256 != plan.compute_sha256():
        raise SlurmBatchRenderError("retry does not bind the persisted resolved plan")
    try:
        validate_enroot_mount_path(plan.selected_profile.profile.workspace_root)
    except ValueError as error:
        raise SlurmBatchRenderError("retry workspace cannot be represented as a safe Enroot mount") from error
    planned_by_id = {shard.shard_id: shard for shard in plan.shards}
    for retry_shard in retry.planned_shards:
        planned = planned_by_id.get(retry_shard.shard_id)
        if planned is None or planned.array_task_index != retry_shard.array_task_index:
            raise SlurmBatchRenderError("retry shard does not match the resolved plan")

    array_tasks = ",".join(str(shard.array_task_index) for shard in retry.planned_shards)
    if plan.array_tasks.max_concurrent is not None:
        array_tasks = f"{array_tasks}%{plan.array_tasks.max_concurrent}"
    directives = render_batch_directives(_build_generation_directives(plan, array=array_tasks))
    attempt_cases = "\n".join(
        f"    {shard.array_task_index}) DD_ATTEMPT_ORDINAL={quote_shell_value(f'{shard.attempt_ordinal:04d}')} ;;"
        for shard in retry.planned_shards
    )
    return f"""#!/usr/bin/env bash
{directives}
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

readonly DD_RUNTIME_ARCHIVE={quote_shell_value(plan.runtime_bundle.path)}
readonly DD_RUNTIME_SHA256={quote_shell_value(plan.runtime_bundle.sha256)}
readonly DD_PLAN={quote_shell_value(plan_path)}
readonly DD_PLAN_SHA256={quote_shell_value(plan.compute_sha256())}
readonly DD_RUN_ROOT={quote_shell_value(run_root)}
readonly DD_WORKSPACE_ROOT={quote_shell_value(plan.selected_profile.profile.workspace_root)}
readonly DD_RUN_ID={quote_shell_value(plan.run_id)}
readonly DD_CLIENT_IMAGE={quote_shell_value(plan.client.image.path)}
readonly DD_CLIENT_IMAGE_SHA256={quote_shell_value(plan.client.image.sha256)}
readonly DD_EFFECTIVE_RESUME_MODE={quote_shell_value(retry.effective_resume_mode)}

verify_sha256() {{
    local actual_sha256
    actual_sha256="$(sha256sum < "$2")"
    [[ "${{actual_sha256%% *}}" == "$1" ]]
}}

verify_sha256 "${{DD_RUNTIME_SHA256}}" "${{DD_RUNTIME_ARCHIVE}}"
verify_sha256 "${{DD_PLAN_SHA256}}" "${{DD_PLAN}}"
verify_sha256 "${{DD_CLIENT_IMAGE_SHA256}}" "${{DD_CLIENT_IMAGE}}"
if [[ ! ${{SLURM_ARRAY_TASK_ID:-}} =~ ^[0-9]+$ ]]; then
    printf '%s\\n' 'SLURM_ARRAY_TASK_ID must be a non-negative integer' >&2
    exit 64
fi
readonly DD_ARRAY_TASK_ID="${{SLURM_ARRAY_TASK_ID}}"
case "${{DD_ARRAY_TASK_ID}}" in
{attempt_cases}
    *) printf '%s\\n' 'array task is absent from retry plan' >&2; exit 64 ;;
esac
readonly DD_ATTEMPT_ORDINAL
printf -v DD_SHARD_ID 'shard-%05d' "${{DD_ARRAY_TASK_ID}}"
readonly DD_SHARD_ID
readonly DD_ATTEMPT_DIR="${{DD_RUN_ROOT}}/shards/${{DD_SHARD_ID}}/attempts/attempt-${{DD_ATTEMPT_ORDINAL}}"
readonly DD_ATTEMPT_MANIFEST="${{DD_ATTEMPT_DIR}}/attempt.json"
for ((DD_WAIT_COUNT = 0; DD_WAIT_COUNT < 300; DD_WAIT_COUNT++)); do
    [[ -f "${{DD_ATTEMPT_MANIFEST}}" ]] && break
    sleep 1
done
if [[ ! -f "${{DD_ATTEMPT_MANIFEST}}" ]]; then
    printf '%s\\n' 'retry attempt state was not published before allocation startup' >&2
    exit 70
fi
if [[ ! ${{SLURM_ARRAY_JOB_ID:-}} =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\\n' 'SLURM_ARRAY_JOB_ID must be a positive integer' >&2
    exit 64
fi
readonly DD_ARRAY_JOB_ID="${{SLURM_ARRAY_JOB_ID}}"
readonly DD_ATTEMPT_ID="attempt-${{DD_ATTEMPT_ORDINAL}}"
enroot start --root --mount "${{DD_WORKSPACE_ROOT}}:${{DD_WORKSPACE_ROOT}}" "${{DD_CLIENT_IMAGE}}" \\
    python -m data_designer.slurm.state.attempt_identity \\
    --workspace-root "${{DD_WORKSPACE_ROOT}}" --run-id "${{DD_RUN_ID}}" \\
    --shard-id "${{DD_SHARD_ID}}" --attempt-id "${{DD_ATTEMPT_ID}}" \\
    --array-job-id "${{DD_ARRAY_JOB_ID}}" --array-task-id "${{DD_ARRAY_TASK_ID}}"
DD_RUNTIME_DIR="$(mktemp -d "${{DD_ATTEMPT_DIR}}/runtime.${{DD_RUNTIME_SHA256}}.XXXXXX")"
readonly DD_RUNTIME_DIR
tar -xzf "${{DD_RUNTIME_ARCHIVE}}" -C "${{DD_RUNTIME_DIR}}"

source "${{DD_RUNTIME_DIR}}/entrypoint.sh"
dd_slurm_run_allocation "${{DD_PLAN}}" "${{DD_ATTEMPT_DIR}}"
"""


def _build_generation_directives(
    plan: ResolvedSlurmRunPlan,
    *,
    array: str | None = None,
) -> tuple[tuple[str, str | None], ...]:
    node_indices = (
        plan.client.host_node_index,
        *(index for deployment in plan.deployments for index in deployment.node_indices),
    )
    node_count = max(node_indices) + 1
    resolved_array = array or "0"
    if array is None and plan.array_tasks.count > 1:
        resolved_array = f"0-{plan.array_tasks.count - 1}"
        if plan.array_tasks.max_concurrent is not None:
            resolved_array = f"{resolved_array}%{plan.array_tasks.max_concurrent}"

    values: list[tuple[str, str | None]] = [
        ("job-name", plan.submission.job_name),
        ("account", plan.submission.account),
        ("partition", plan.submission.partition),
        ("nodes", str(node_count)),
        ("cpus-per-task", str(plan.client.authored.cpus)),
        ("time", plan.submission.time_limit),
        ("array", resolved_array),
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
