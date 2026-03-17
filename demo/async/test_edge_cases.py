# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test: edge cases for the async scheduler.

Sub-tests:
1. Minimum generation (num_records=1) — produces correct dataset and traces.
2. Tiny buffer_size (buffer_size=1) — multiple row groups schedule correctly.
3. Early shutdown — failing column triggers error-rate shutdown; traces capture errors.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"

from helpers import (
    check,
    create_base_config,
    create_data_designer,
    create_mock_model_config,
    patch_llm_responses,
    seed_rng,
)

from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.run_config import RunConfig


def _simple_config():
    config = create_base_config()
    config.add_column(
        LLMTextColumnConfig(
            name="output",
            model_alias="openai-text",
            prompt="Process seed {{ seed_value }}.",
        )
    )
    return config


def _config_with_fragile_column():
    """Config where 'fragile' column prompt triggers mock failures."""
    config = create_base_config()
    config.add_column(
        LLMTextColumnConfig(
            name="stable",
            model_alias="openai-text",
            prompt="Process seed {{ seed_value }}.",
        )
    )
    config.add_column(
        LLMTextColumnConfig(
            name="fragile",
            model_alias="openai-text",
            prompt="FRAGILE_MARKER: analyze {{ seed_value }}.",
        )
    )
    return config


# ---------------------------------------------------------------------------
# Sub-test 1: minimum generation
# ---------------------------------------------------------------------------


def test_single_record() -> None:
    print("\n--- Sub-test 1: Single record (num_records=1) ---")

    with tempfile.TemporaryDirectory() as artifact_dir:
        seed_rng(42)
        dd = create_data_designer(artifact_path=artifact_dir, async_trace=True)
        config = _simple_config()

        with patch_llm_responses():
            result = dd.create(config, num_records=1, dataset_name="single")

        df = result.load_dataset()
        traces = result.task_traces

        print(f"  rows={len(df)}, cols={len(df.columns)}, traces={len(traces)}")

        check(len(df) == 1, "Single row produced")
        check("seed_value" in df.columns, "seed_value column present")
        check("output" in df.columns, "output column present")
        check(len(traces) > 0, f"Traces captured ({len(traces)})")
        check(
            all(t.status == "ok" for t in traces),
            "All traces succeeded",
        )


# ---------------------------------------------------------------------------
# Sub-test 2: tiny buffer_size
# ---------------------------------------------------------------------------


def test_tiny_buffer() -> None:
    # buffer_size=1 with 8 records creates 8 row groups, well above
    # the default max_concurrent_row_groups=3.  This exercises the
    # scheduler's streaming admission (row groups admitted as slots free).
    num_records = 8
    print(f"\n--- Sub-test 2: Tiny buffer_size (buffer_size=1, num_records={num_records}) ---")

    with tempfile.TemporaryDirectory() as artifact_dir:
        seed_rng(42)
        _, provider = create_mock_model_config()

        from data_designer.interface import DataDesigner

        dd = DataDesigner(artifact_path=artifact_dir, model_providers=[provider])
        dd.set_run_config(
            RunConfig(
                buffer_size=1,
                disable_early_shutdown=True,
                async_trace=True,
            )
        )

        config = _simple_config()

        with patch_llm_responses():
            result = dd.create(config, num_records=num_records, dataset_name="tiny-buffer")

        df = result.load_dataset()
        traces = result.task_traces

        row_groups_seen = {t.row_group for t in traces}

        print(f"  rows={len(df)}, traces={len(traces)}, row_groups={sorted(row_groups_seen)}")

        check(len(df) == num_records, f"All {num_records} rows produced")
        check(len(row_groups_seen) == num_records, f"Got {num_records} distinct row groups")
        check(
            all(t.status == "ok" for t in traces),
            "All traces succeeded",
        )


# ---------------------------------------------------------------------------
# Sub-test 3: early shutdown via error rate
# ---------------------------------------------------------------------------


def test_early_shutdown() -> None:
    print("\n--- Sub-test 3: Early shutdown from error rate ---")

    with tempfile.TemporaryDirectory() as artifact_dir:
        seed_rng(42)
        _, provider = create_mock_model_config()

        from data_designer.interface import DataDesigner
        from data_designer.interface.errors import DataDesignerGenerationError, DataDesignerProfilingError

        dd = DataDesigner(artifact_path=artifact_dir, model_providers=[provider])
        dd.set_run_config(
            RunConfig(
                buffer_size=1000,
                disable_early_shutdown=False,
                shutdown_error_rate=0.3,
                shutdown_error_window=5,
                async_trace=True,
            )
        )

        # Capture the builder so we can read traces even if profiling fails
        # (early shutdown may produce no data → no parquet → profiling error).
        captured_builder = None
        original_create = dd._create_dataset_builder

        def _capture_builder(config, rp):
            nonlocal captured_builder
            captured_builder = original_create(config, rp)
            return captured_builder

        dd._create_dataset_builder = _capture_builder

        config = _config_with_fragile_column()
        num_records = 16

        try:
            with patch_llm_responses(fail_pattern="FRAGILE_MARKER", fail_rate=1.0):
                result = dd.create(config, num_records=num_records, dataset_name="shutdown")
            traces = result.task_traces
        except (DataDesignerGenerationError, DataDesignerProfilingError):
            # Early shutdown with 100% failures leaves no data — profiling fails.
            traces = captured_builder.task_traces if captured_builder else []
        finally:
            dd._create_dataset_builder = original_create

        error_traces = [t for t in traces if t.status == "error"]
        ok_traces = [t for t in traces if t.status == "ok"]

        print(f"  traces={len(traces)}, ok={len(ok_traces)}, errors={len(error_traces)}")

        check(len(traces) > 0, "Traces were captured")
        check(len(error_traces) > 0, f"Error traces present ({len(error_traces)})")
        check(
            any(t.column == "fragile" for t in error_traces),
            "Errors are on the 'fragile' column",
        )

        # With 100% fail rate on fragile column and low error window,
        # the scheduler should have stopped before completing everything.
        total_expected = num_records * 3  # seed_value + stable + fragile, each cell-by-cell
        check(
            len(traces) < total_expected,
            f"Early shutdown: {len(traces)} traces < {total_expected} expected",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Test: Edge Cases")
    print("=" * 60)

    test_single_record()
    test_tiny_buffer()
    test_early_shutdown()

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        sys.exit(1)
