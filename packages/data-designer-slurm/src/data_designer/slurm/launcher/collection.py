# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe rendering for zero-GPU CPU collection jobs."""

from __future__ import annotations

import posixpath
from pathlib import PurePosixPath

from data_designer.slurm.launcher.batch import quote_shell_value, render_batch_directives
from data_designer.slurm.launcher.errors import SlurmBatchRenderError
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.destinations import CollectionDestination
from data_designer.slurm.state.outputs import CollectionPlan


def render_collection_script(
    resolved_plan: ResolvedSlurmRunPlan,
    collection_plan: CollectionPlan,
    destination: CollectionDestination,
) -> str:
    """Render one CPU-only job that invokes the allocation-gated collection worker."""
    if collection_plan.run_id != resolved_plan.run_id:
        raise SlurmBatchRenderError("collection run identity does not match the resolved plan")
    if collection_plan.host_destination != destination.host_path:
        raise SlurmBatchRenderError("collection host destination does not match its resolved mount")
    if collection_plan.container_destination != destination.container_path:
        raise SlurmBatchRenderError("collection container destination does not match its resolved mount")

    collection_root = posixpath.join(
        posixpath.dirname(resolved_plan.authored_config.path),
        "collections",
        collection_plan.collection_id,
    )
    collection_plan_path = posixpath.join(collection_root, "plan.json")
    directives = render_batch_directives(
        (
            ("job-name", collection_plan.submission_job_name),
            ("account", resolved_plan.submission.account),
            ("partition", resolved_plan.submission.partition),
            ("nodes", "1"),
            ("ntasks", "1"),
            ("cpus-per-task", str(resolved_plan.client.authored.cpus)),
            ("time", resolved_plan.submission.time_limit),
            ("chdir", collection_root),
            ("output", f"{collection_root}/slurm-%j.out"),
            ("error", f"{collection_root}/slurm-%j.err"),
        )
    )
    workspace_root = resolved_plan.selected_profile.profile.workspace_root
    state_mount = f"{workspace_root}:{workspace_root}"
    output_mount = f"{destination.mount.source}:{destination.mount.target}"
    mount_arguments = _render_mount_arguments(
        ("DD_STATE_MOUNT", workspace_root, workspace_root),
        ("DD_OUTPUT_MOUNT", destination.mount.source, destination.mount.target),
    )
    return f"""#!/usr/bin/env bash
{directives}
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

readonly DD_CLIENT_IMAGE={quote_shell_value(resolved_plan.client.image.path)}
readonly DD_CLIENT_IMAGE_SHA256={quote_shell_value(resolved_plan.client.image.sha256)}
readonly DD_COLLECTION_PLAN={quote_shell_value(collection_plan_path)}
readonly DD_COLLECTION_PLAN_SHA256={quote_shell_value(collection_plan.compute_sha256())}
readonly DD_WORKSPACE_ROOT={quote_shell_value(resolved_plan.selected_profile.profile.workspace_root)}
readonly DD_RUN_ID={quote_shell_value(resolved_plan.run_id)}
readonly DD_COLLECTION_ID={quote_shell_value(collection_plan.collection_id)}
readonly DD_STATE_MOUNT={quote_shell_value(state_mount)}
readonly DD_OUTPUT_MOUNT={quote_shell_value(output_mount)}

verify_sha256() {{
    local actual_sha256
    actual_sha256="$(sha256sum < "$2")"
    [[ "${{actual_sha256%% *}}" == "$1" ]]
}}

verify_sha256 "${{DD_CLIENT_IMAGE_SHA256}}" "${{DD_CLIENT_IMAGE}}"
verify_sha256 "${{DD_COLLECTION_PLAN_SHA256}}" "${{DD_COLLECTION_PLAN}}"
DD_ENROOT_MOUNTS=({mount_arguments})
readonly DD_ENROOT_MOUNTS
exec enroot start --root "${{DD_ENROOT_MOUNTS[@]}}" "${{DD_CLIENT_IMAGE}}" \
    python -m data_designer.slurm.state.collection_worker \
    --workspace-root "${{DD_WORKSPACE_ROOT}}" --run-id "${{DD_RUN_ID}}" --collection-id "${{DD_COLLECTION_ID}}"
"""


def _render_mount_arguments(*mounts: tuple[str, str, str]) -> str:
    unique: dict[str, tuple[str, str, str]] = {}
    targets: dict[str, str] = {}
    for variable, source, target in mounts:
        mount = f"{source}:{target}"
        existing_source = targets.get(target)
        if existing_source is not None and existing_source != source:
            raise SlurmBatchRenderError("collection state and output mounts cannot share a target")
        targets[target] = source
        unique.setdefault(mount, (variable, source, target))
    ordered = sorted(unique.values(), key=lambda item: len(PurePosixPath(item[2]).parts))
    return " ".join(f'--mount "${{{variable}}}"' for variable, _, _ in ordered)


__all__ = ["render_collection_script"]
