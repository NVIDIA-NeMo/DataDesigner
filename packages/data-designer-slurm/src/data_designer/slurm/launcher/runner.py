# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Injectable process execution for Slurm command-line tools."""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Protocol


class CommandRunner(Protocol):
    """Minimal command boundary implemented by production and fake runners."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Execute one argument-vector command."""
        ...


class SubprocessRunner:
    """Run commands without a shell or unrestricted ambient environment."""

    _environment: Mapping[str, str]
    _timeout_seconds: float

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        explicit_environment = (
            dict(environment) if environment is not None else {"PATH": os.environ.get("PATH") or os.defpath}
        )
        for name, value in explicit_environment.items():
            if type(name) is not str or not name or "=" in name or "\0" in name:
                raise ValueError("environment names must be non-empty and must not contain '=' or NUL")
            if type(value) is not str or "\0" in value:
                raise ValueError("environment values must not contain NUL")
        self._environment = MappingProxyType({**explicit_environment, "LC_ALL": "C"})
        self._timeout_seconds = float(timeout_seconds)

    @property
    def environment(self) -> Mapping[str, str]:
        """Return the allowlisted environment forwarded to child processes."""
        return self._environment

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Execute an argument vector with captured text output."""
        return subprocess.run(
            tuple(command),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=dict(self._environment),
            timeout=self._timeout_seconds,
        )
