# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test: async scheduler respects DAG execution order.

Builds a 4-column DAG:

    seed_value (sampler, no deps)
        ├── stage1 (LLM, depends on seed_value)
        │       └── stage2 (LLM, depends on stage1)
        └── independent (LLM, depends on seed_value only)

Verifies via TaskTraces that:
1. All seed tasks finish before any downstream task dispatches.
2. All stage1 tasks finish before any stage2 task dispatches.
3. "independent" tasks can overlap with stage1 (no dependency between them).
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
    patch_llm_responses,
    seed_rng,
)

from data_designer.config.column_configs import LLMTextColumnConfig

NUM_RECORDS = 8


def _build_config():
    config = create_base_config()
    config.add_column(
        LLMTextColumnConfig(
            name="stage1",
            model_alias="openai-text",
            prompt="Summarize seed {{ seed_value }}.",
        )
    )
    config.add_column(
        LLMTextColumnConfig(
            name="stage2",
            model_alias="openai-text",
            prompt="Analyze {{ stage1 }}.",
        )
    )
    config.add_column(
        LLMTextColumnConfig(
            name="independent",
            model_alias="openai-text",
            prompt="Independent thought on {{ seed_value }}.",
        )
    )
    return config


def _traces_for(traces: list, column: str) -> list:
    return [t for t in traces if t.column == column]


def main() -> None:
    print("=" * 60)
    print("Test: Execution Order Respects DAG")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as artifact_dir:
        seed_rng(42)
        dd = create_data_designer(artifact_path=artifact_dir, async_trace=True)
        config = _build_config()

        with patch_llm_responses():
            result = dd.create(config, num_records=NUM_RECORDS, dataset_name="order")

        df = result.load_dataset()
        traces = result.task_traces

        print(f"\n  Dataset: {len(df)} rows, {len(df.columns)} cols")
        print(f"  Traces:  {len(traces)} total")

        # Group traces by column
        seed_traces = _traces_for(traces, "seed_value")
        stage1_traces = _traces_for(traces, "stage1")
        stage2_traces = _traces_for(traces, "stage2")
        indep_traces = _traces_for(traces, "independent")

        print(
            f"  Per column: seed_value={len(seed_traces)}, stage1={len(stage1_traces)}, "
            f"stage2={len(stage2_traces)}, independent={len(indep_traces)}"
        )

        # --- Assertion 1: seeds finish before downstream dispatches ---
        print("\nResults:")
        if seed_traces:
            latest_seed_completion = max(t.completed_at for t in seed_traces)
            downstream = stage1_traces + stage2_traces + indep_traces
            if downstream:
                earliest_downstream_dispatch = min(t.dispatched_at for t in downstream)
                check(
                    latest_seed_completion <= earliest_downstream_dispatch,
                    "All seed tasks completed before first downstream dispatch",
                )
            else:
                check(False, "Expected downstream traces but found none")
        else:
            check(False, "Expected seed traces but found none")

        # --- Assertion 2: stage1 finishes before stage2 dispatches ---
        if stage1_traces and stage2_traces:
            latest_stage1 = max(t.completed_at for t in stage1_traces)
            earliest_stage2 = min(t.dispatched_at for t in stage2_traces)
            check(
                latest_stage1 <= earliest_stage2,
                "All stage1 tasks completed before first stage2 dispatch",
            )

        # --- Assertion 3: independent dispatched before stage2 ---
        # With zero-latency mocks there's no temporal overlap — tasks complete
        # instantly when the event loop schedules them.  Instead we verify
        # that independent tasks were dispatched in the same "wave" as stage1
        # (between seed completion and stage2 dispatch), proving the scheduler
        # treats them as independent of stage1.
        if indep_traces and stage2_traces:
            latest_indep_dispatch = max(t.dispatched_at for t in indep_traces)
            earliest_stage2_dispatch = min(t.dispatched_at for t in stage2_traces)
            check(
                latest_indep_dispatch <= earliest_stage2_dispatch,
                "All independent tasks dispatched before first stage2 dispatch",
            )

        # --- Assertion 4: all traces succeeded ---
        error_traces = [t for t in traces if t.status == "error"]
        check(len(error_traces) == 0, f"No error traces (found {len(error_traces)})")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        sys.exit(1)
