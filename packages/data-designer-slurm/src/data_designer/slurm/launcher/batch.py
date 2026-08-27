# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared safe rendering primitives for package-owned Slurm batch scripts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from data_designer.slurm.launcher.errors import SlurmBatchRenderError

_DIRECTIVE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_DIRECTIVE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/,%+-]*$")


@dataclass(frozen=True)
class BatchDirective:
    """One validated ``#SBATCH`` option."""

    name: str
    value: str

    def render(self) -> str:
        """Render the directive as one non-executable scheduler line."""
        if type(self.name) is not str or _DIRECTIVE_NAME_PATTERN.fullmatch(self.name) is None:
            raise SlurmBatchRenderError("batch directive name is invalid")
        if type(self.value) is not str:
            raise SlurmBatchRenderError("batch directive value must be text")
        reject_control_characters(self.value, field_name=f"--{self.name} value")
        value = self.value if _DIRECTIVE_TOKEN_PATTERN.fullmatch(self.value) else _quote_sbatch_option_value(self.value)
        return f"#SBATCH --{self.name}={value}"


def render_batch_directives(values: tuple[tuple[str, str | None], ...]) -> str:
    """Render ordered optional directive values as safe scheduler lines."""
    return "\n".join(BatchDirective(name=name, value=value).render() for name, value in values if value is not None)


def quote_shell_value(value: str) -> str:
    """Quote one validated value for literal use in package-owned Bash."""
    if type(value) is not str:
        raise SlurmBatchRenderError("shell value must be text")
    reject_control_characters(value, field_name="shell value")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def reject_control_characters(value: str, *, field_name: str) -> None:
    """Reject characters that can split a directive or shell assignment."""
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SlurmBatchRenderError(f"{field_name} must not contain control characters")


def _quote_sbatch_option_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
