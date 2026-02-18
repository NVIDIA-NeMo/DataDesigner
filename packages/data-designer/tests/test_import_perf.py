# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Baseline performance measurements (for reference):
#   import_only (pure Python import, no CLI):
#     Cold:  0.091s
#     Warm:  0.036s mean, 0.034s median, 0.008s stdev [0.033s - 0.058s]
#
#   cli_help (import + argparse setup):
#     Cold:  1.171s
#     Warm:  0.142s mean, 0.093s median, 0.107s stdev [0.089s - 0.405s]
#
#   config_list (import + config loading):
#     Cold:  3.161s
#     Warm:  0.554s mean, 0.267s median, 0.737s stdev [0.251s - 2.619s]
#
#   perf-import: 0.008 - 0.02s
MAX_IMPORT_TIME_SECONDS = 3.0
PERF_TEST_TIMEOUT_SECONDS = 10.0


def test_import_performance() -> None:
    """Test that average pure import time never exceeds 6 seconds (1 cold start + 4 warm cache runs)."""
    # Get the project root (where Makefile is located)
    # For workspace packages, need to go up to the workspace root
    project_root = Path(__file__).parent.parent.parent.parent

    num_runs = 5
    import_times = []

    for run in range(num_runs):
        # Clean cache only on first run (cold start), rest use warm cache
        cmd = ["make", "perf-import", "NOFILE=1"]
        if run == 0:
            cmd.append("CLEAN=1")

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=PERF_TEST_TIMEOUT_SECONDS,
        )

        assert result.returncode == 0, f"perf-import failed on run {run + 1}:\n{result.stdout}\n{result.stderr}"

        # Parse the output to extract pure-import time
        # Looking for line like: "  Total: 3.456s"
        match = re.search(r"Total:\s+([\d.]+)s", result.stdout)
        assert match, f"Could not parse import time from run {run + 1}:\n{result.stdout}"

        import_time = float(match.group(1))
        import_times.append(import_time)

    # Calculate average
    avg_import_time = sum(import_times) / len(import_times)
    min_import_time = min(import_times)
    max_import_time = max(import_times)

    # Print summary for debugging
    print("\nImport Performance Summary:")
    print(f"  Runs: {num_runs} (1 cold start + {num_runs - 1} warm cache)")
    print(f"  Cold start (run 1): {import_times[0]:.3f}s")
    print(f"  Warm cache (runs 2-{num_runs}): {', '.join(f'{t:.3f}s' for t in import_times[1:])}")
    print(f"  Average: {avg_import_time:.3f}s")
    print(f"  Min: {min_import_time:.3f}s")
    print(f"  Max: {max_import_time:.3f}s")

    # Assert average import time is under threshold
    assert avg_import_time < MAX_IMPORT_TIME_SECONDS, (
        f"Average import time {avg_import_time:.3f}s exceeds {MAX_IMPORT_TIME_SECONDS}s threshold "
        f"(times: {import_times})"
    )
