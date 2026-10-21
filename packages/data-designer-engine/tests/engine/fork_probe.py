# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run fork-safety scenarios in a fresh interpreter.

Forking the pytest worker copies everything earlier tests left behind (DuckDB
thread pools, large allocator reservations), which makes fork() itself fail
with ENOMEM on memory-constrained CI runners. A fresh interpreter gives each
scenario a small, clean parent.
"""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import textwrap

import pytest

requires_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="requires the fork start method"
)

_CHILD_JOIN_TIMEOUT_S = 20
_SCRIPT_TIMEOUT_S = 90

_PRELUDE = f"""
import asyncio
import multiprocessing
import sys


async def ping():
    return "pong"


def run_forked_child(target):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=target, args=(queue,))
    child.start()
    child.join(timeout={_CHILD_JOIN_TIMEOUT_S})
    if child.is_alive():
        child.kill()
        child.join()
        sys.exit("forked child hung")
    if child.exitcode != 0:
        sys.exit(f"forked child exited with code {{child.exitcode}}")
    return queue.get(timeout=5)
"""


def run_fork_probe(body: str) -> None:
    """Run ``body`` after a prelude that defines ``ping()`` and ``run_forked_child(target)``.

    ``target`` receives a queue and must put its result on it. The probe fails if
    the script exits non-zero, which includes a forked child that hangs.
    """
    script = _PRELUDE + textwrap.dedent(body)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=_SCRIPT_TIMEOUT_S)
    assert result.returncode == 0, f"fork probe failed:\n{result.stderr}"
