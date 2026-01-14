# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
from pathlib import Path

# Maximum allowed average import time in seconds
# Should be < 5 but leaving some room for CI variability
MAX_IMPORT_TIME_SECONDS = 7.0
PERF_TEST_TIMEOUT_SECONDS = 15.0


def test_import_performance():
    """Test that average import time never exceeds MAX_IMPORT_TIME_SECONDS (with clean cache)."""
    # Get the project root (where Makefile is located)
    project_root = Path(__file__).parent.parent

    num_runs = 5
    import_times = []

    for run in range(num_runs):
        # Run make perf-import with clean cache and no file output
        result = subprocess.run(
            ["make", "perf-import", "CLEAN=1", "NOFILE=1"],
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
    print(f"  Runs: {num_runs}")
    print(f"  Times: {', '.join(f'{t:.3f}s' for t in import_times)}")
    print(f"  Average: {avg_import_time:.3f}s")
    print(f"  Min: {min_import_time:.3f}s")
    print(f"  Max: {max_import_time:.3f}s")

    # Assert average import time is under threshold
    assert avg_import_time < MAX_IMPORT_TIME_SECONDS, (
        f"Average import time {avg_import_time:.3f}s exceeds {MAX_IMPORT_TIME_SECONDS}s threshold "
        f"(times: {import_times})"
    )
