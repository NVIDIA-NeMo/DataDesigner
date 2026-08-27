# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal safe rendering for thin deterministic Slurm batch entrypoints."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass

from data_designer.slurm.launcher.errors import SlurmBatchRenderError
from data_designer.slurm.planning import ResolvedSlurmRunPlan

_DIRECTIVE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_DIRECTIVE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/,%+-]*$")


@dataclass(frozen=True, slots=True)
class _BatchDirective:
    """One validated ``#SBATCH`` option."""

    name: str
    value: str

    def render(self) -> str:
        """Render the directive as one non-executable scheduler line."""
        if type(self.name) is not str or _DIRECTIVE_NAME_PATTERN.fullmatch(self.name) is None:
            raise SlurmBatchRenderError("batch directive name is invalid")
        if type(self.value) is not str:
            raise SlurmBatchRenderError("batch directive value must be text")
        _reject_control_characters(self.value, field_name=f"--{self.name} value")
        value = self.value if _DIRECTIVE_TOKEN_PATTERN.fullmatch(self.value) else _quote_sbatch_option_value(self.value)
        return f"#SBATCH --{self.name}={value}"


def render_generation_attempt_script(plan: ResolvedSlurmRunPlan, *, attempt_ordinal: int) -> str:
    """Render a resolved generation plan as one thin deterministic entrypoint."""
    if type(attempt_ordinal) is not int or attempt_ordinal <= 0:
        raise SlurmBatchRenderError("attempt_ordinal must be a positive integer")

    run_root = posixpath.dirname(plan.authored_config.path)
    plan_path = posixpath.join(run_root, "resolved-plan.json")
    directives = _build_generation_directives(plan)
    directive_text = "\n".join(directive.render() for directive in directives)
    attempt = f"{attempt_ordinal:04d}"

    return f"""#!/usr/bin/env bash
{directive_text}
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

readonly DD_RUNTIME_ARCHIVE={_quote_shell_value(plan.runtime_bundle.path)}
readonly DD_RUNTIME_SHA256={_quote_shell_value(plan.runtime_bundle.sha256)}
readonly DD_PLAN={_quote_shell_value(plan_path)}
readonly DD_PLAN_SHA256={_quote_shell_value(plan.compute_sha256())}
readonly DD_RUN_ROOT={_quote_shell_value(run_root)}
readonly DD_ATTEMPT_ORDINAL={_quote_shell_value(attempt)}

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


def _build_generation_directives(plan: ResolvedSlurmRunPlan) -> tuple[_BatchDirective, ...]:
    node_indices = (
        plan.client.host_node_index,
        *(index for deployment in plan.deployments for index in deployment.node_indices),
    )
    node_count = max(node_indices) + 1
    array = "0"
    if plan.array_tasks.count > 1:
        array = f"0-{plan.array_tasks.count - 1}%{plan.array_tasks.max_concurrent}"

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
        # TODO(#875): Remove this defense once config and plan validation make this state unrepresentable.
        raise SlurmBatchRenderError("mem_per_gpu requires GRES GPU request mode")
    if profile.scheduler.mem_per_gpu is not None:
        values.append(("mem-per-gpu", profile.scheduler.mem_per_gpu))
    if plan.submission.comment is not None:
        values.append(("comment", plan.submission.comment))
    return tuple(_BatchDirective(name=name, value=value) for name, value in values if value is not None)


def _quote_sbatch_option_value(value: str) -> str:
    _reject_control_characters(value, field_name="batch directive value")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _quote_shell_value(value: str) -> str:
    _reject_control_characters(value, field_name="shell value")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def _reject_control_characters(value: str, *, field_name: str) -> None:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SlurmBatchRenderError(f"{field_name} must not contain control characters")
