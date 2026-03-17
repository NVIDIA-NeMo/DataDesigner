# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test: sync builder vs async scheduler produce structurally identical datasets.

Uses real LLMs (openai-text). Compares:
- Row / column counts
- Non-null values in every cell
- TaskTrace availability (async only)
- Wall-clock timing
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Enable async engine before any data_designer imports.
os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"

from helpers import check, seed_rng

from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.interface import DataDesigner

NUM_RECORDS = 5


def _build_config() -> DataDesignerConfigBuilder:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="topic",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["science", "history", "art"]),
        )
    )
    config.add_column(
        LLMTextColumnConfig(
            name="summary",
            model_alias="openai-text",
            prompt="Write a one-sentence summary about {{ topic }}.",
        )
    )
    config.add_column(
        LLMTextColumnConfig(
            name="followup",
            model_alias="openai-text",
            prompt="Given this summary: '{{ summary }}', write a follow-up question.",
        )
    )
    return config


def _print_trace_table(traces: list) -> None:
    if not traces:
        print("  (no traces)")
        return

    header = f"  {'column':<12} {'rg':>3} {'row':>4} {'type':<13} {'wait':>7} {'run':>7} {'status':<6} {'error'}"
    print(header)
    print(f"  {'-' * len(header.strip())}")

    for t in sorted(traces, key=lambda t: t.dispatched_at):
        wait_ms = (t.slot_acquired_at - t.dispatched_at) * 1000 if t.slot_acquired_at and t.dispatched_at else 0
        run_ms = (t.completed_at - t.slot_acquired_at) * 1000 if t.completed_at and t.slot_acquired_at else 0
        row_str = str(t.row_index) if t.row_index is not None else "-"
        err_str = (t.error or "")[:40]
        print(
            f"  {t.column:<12} {t.row_group:>3} {row_str:>4} {t.task_type:<13} "
            f"{wait_ms:>6.0f}ms {run_ms:>6.0f}ms {t.status:<6} {err_str}"
        )


def main() -> None:
    print("=" * 60)
    print("Test: Sync vs Async Parity (real LLMs)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as async_dir, tempfile.TemporaryDirectory() as sync_dir:
        # --- Sync run first (to warm up health checks, caches, etc.) ---
        import data_designer.engine.dataset_builders.column_wise_builder as cwb

        cwb.DATA_DESIGNER_ASYNC_ENGINE = False

        print(f"\n[1/2] Running sync builder ({NUM_RECORDS} records)...")
        seed_rng(42)
        dd_sync = DataDesigner(artifact_path=sync_dir)
        dd_sync.set_run_config(RunConfig(buffer_size=1000, disable_early_shutdown=True, async_trace=False))
        config_sync = _build_config()

        t0 = time.perf_counter()
        result_sync = dd_sync.create(config_sync, num_records=NUM_RECORDS, dataset_name="sync")
        sync_time = time.perf_counter() - t0

        df_sync = result_sync.load_dataset()

        print(f"  rows={len(df_sync)}, cols={len(df_sync.columns)}, time={sync_time:.3f}s")

        # --- Async run second ---
        cwb.DATA_DESIGNER_ASYNC_ENGINE = True

        print(f"\n[2/2] Running async scheduler ({NUM_RECORDS} records)...")
        seed_rng(42)
        dd_async = DataDesigner(artifact_path=async_dir)
        dd_async.set_run_config(RunConfig(buffer_size=1000, disable_early_shutdown=True, async_trace=True))
        config_async = _build_config()

        t0 = time.perf_counter()
        result_async = dd_async.create(config_async, num_records=NUM_RECORDS, dataset_name="async")
        async_time = time.perf_counter() - t0

        df_async = result_async.load_dataset()
        traces_async = result_async.task_traces

        print(
            f"  rows={len(df_async)}, cols={len(df_async.columns)}, time={async_time:.3f}s, traces={len(traces_async)}"
        )

        print("\n  Async traces:")
        _print_trace_table(traces_async)

        # --- Assertions ---
        print("\nResults:")
        check(len(df_async) == NUM_RECORDS, f"Async produced {NUM_RECORDS} rows")
        check(len(df_sync) == NUM_RECORDS, f"Sync produced {NUM_RECORDS} rows")
        check(len(df_async) == len(df_sync), f"Row counts match ({len(df_async)})")
        check(
            set(df_async.columns) == set(df_sync.columns),
            f"Column sets match ({sorted(df_async.columns)})",
        )
        check(
            df_async.notna().all().all(),
            "Async: no null values",
        )
        check(
            df_sync.notna().all().all(),
            "Sync: no null values",
        )
        check(len(traces_async) > 0, f"Async produced {len(traces_async)} TaskTraces")
        check(len(result_sync.task_traces) == 0, "Sync produced no TaskTraces")

        speedup = sync_time / async_time if async_time > 0 else float("inf")
        print(f"\n  Timing: sync={sync_time:.3f}s  async={async_time:.3f}s  ratio={speedup:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        sys.exit(1)
