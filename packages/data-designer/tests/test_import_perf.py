# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
from pathlib import Path

# Maximum allowed average import time in seconds
# Average of 1 cold start + 4 warm cache runs
# Cold starts vary 4-13s due to OS caching, system load, CPU scaling
# Warm cache consistently <3s. Average should be well under 6s.
MAX_IMPORT_TIME_SECONDS = 6.0
PERF_TEST_TIMEOUT_SECONDS = 30.0


def test_import_performance():
    """Test that average import time never exceeds 6 seconds (1 cold start + 4 warm cache runs)."""
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

        # Parse the output to extract import time
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
