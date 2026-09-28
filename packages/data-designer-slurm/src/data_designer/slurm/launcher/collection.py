# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe rendering for zero-GPU CPU collection jobs."""

from __future__ import annotations

import posixpath
from importlib import resources

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
    if collection_plan.python_executable is None:
        raise SlurmBatchRenderError("native collection requires a v2 plan with a Python executable")

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
            ("partition", resolved_plan.selected_profile.profile.image_build.partition),
            ("nodes", "1"),
            ("ntasks", "1"),
            ("cpus-per-task", str(resolved_plan.client.authored.cpus)),
            ("time", resolved_plan.submission.time_limit),
            ("chdir", collection_root),
            ("output", f"{collection_root}/slurm-%j.out"),
            ("error", f"{collection_root}/slurm-%j.err"),
        )
    )
    scratch_source = resources.files("data_designer.slurm.runtime").joinpath("scratch.sh").read_text()
    return f"""#!/usr/bin/env bash
{directives}
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONNOUSERSITE=1

{scratch_source}
readonly DD_COLLECTION_PLAN={quote_shell_value(collection_plan_path)}
readonly DD_COLLECTION_PLAN_SHA256={quote_shell_value(collection_plan.compute_sha256())}
readonly DD_COLLECTION_PYTHON={quote_shell_value(collection_plan.python_executable)}
readonly DD_PACKAGE_VERSION={quote_shell_value(resolved_plan.package_version)}
readonly DD_RUNTIME_ARCHIVE={quote_shell_value(resolved_plan.runtime_bundle.path)}
readonly DD_RUNTIME_SHA256={quote_shell_value(resolved_plan.runtime_bundle.sha256)}
readonly DD_WORKSPACE_ROOT={quote_shell_value(resolved_plan.selected_profile.profile.workspace_root)}
readonly DD_RUN_ID={quote_shell_value(resolved_plan.run_id)}
readonly DD_COLLECTION_ID={quote_shell_value(collection_plan.collection_id)}

verify_sha256() {{
    local actual_sha256
    actual_sha256="$(sha256sum < "$2")"
    [[ "${{actual_sha256%% *}}" == "$1" ]]
}}

verify_sha256 "${{DD_COLLECTION_PLAN_SHA256}}" "${{DD_COLLECTION_PLAN}}"
[[ -x "${{DD_COLLECTION_PYTHON}}" ]]
actual_package_version="$("${{DD_COLLECTION_PYTHON}}" -c \
    'from importlib.metadata import version; print(version("data-designer-slurm"))')"
[[ "${{actual_package_version}}" == "${{DD_PACKAGE_VERSION}}" ]]
trap 'status=$?; dd_remove_local_scratch "${{DD_SCRATCH_ROOT}}" || true; exit "${{status}}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
dd_initialize_local_scratch
dd_extract_runtime_bundle "${{DD_RUNTIME_ARCHIVE}}" "${{DD_RUNTIME_SHA256}}"
export PYTHONPATH="${{DD_SCRATCH_ROOT}}/runtime"
"${{DD_COLLECTION_PYTHON}}" -m data_designer.slurm.state.collection_worker \
    --workspace-root "${{DD_WORKSPACE_ROOT}}" --run-id "${{DD_RUN_ID}}" --collection-id "${{DD_COLLECTION_ID}}"
"""


__all__ = ["render_collection_script"]
