# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap-safe process boundary for the plugin-aware client worker.

This module must remain free of Data Designer configuration, interface, plugin,
and worker imports. The child process activates its verified dependency overlay
before importing any of those modules.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

ProcessExecutor = Callable[[tuple[str, ...]], int]


def execute_process(command: tuple[str, ...]) -> int:
    """Execute a child process and return its exit status."""
    return subprocess.run(command, check=False).returncode


@dataclass(frozen=True)
class ClientWorkerProcess:
    """Launch the plugin-aware client worker in a fresh Python interpreter."""

    executable: str = field(default_factory=lambda: sys.executable)
    executor: ProcessExecutor = execute_process

    def run(self, arguments: Sequence[str]) -> int:
        """Run one client-worker operation without inheriting imported modules."""
        command = (self.executable, "-m", "data_designer.slurm.client.worker", *arguments)
        return self.executor(command)
