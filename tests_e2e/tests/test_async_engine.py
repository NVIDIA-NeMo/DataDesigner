# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest

import data_designer.config as dd
import data_designer.engine.dataset_builders.column_wise_builder as cwb
from data_designer.config.run_config import RunConfig
from data_designer.interface import DataDesigner

NUM_RECORDS = 2
PARALLEL_COLUMNS = ("summary", "analysis")


def test_async_engine_concurrent_columns(tmp_path: Path) -> None:
    """Verify the async engine runs independent LLM columns concurrently."""
    if os.environ.get("NVIDIA_API_KEY") is None:
        pytest.skip("NVIDIA_API_KEY must be set")

    original = cwb.DATA_DESIGNER_ASYNC_ENGINE
    cwb.DATA_DESIGNER_ASYNC_ENGINE = True
    try:
        dd_instance = DataDesigner(artifact_path=str(tmp_path))
        dd_instance.set_run_config(RunConfig(buffer_size=NUM_RECORDS, async_trace=True))

        config = dd.DataDesignerConfigBuilder()
        config.add_column(
            dd.SamplerColumnConfig(
                name="topic",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=["science", "history", "art"]),
            )
        )
        for col in PARALLEL_COLUMNS:
            config.add_column(
                dd.LLMTextColumnConfig(
                    name=col,
                    model_alias="nvidia-text",
                    prompt=f"Write one sentence about {{{{ topic }}}} ({col}).",
                )
            )

        result = dd_instance.create(config, num_records=NUM_RECORDS, dataset_name="async_e2e")
        df = result.load_dataset()
    finally:
        cwb.DATA_DESIGNER_ASYNC_ENGINE = original

    # Dataset correctness
    assert len(df) == NUM_RECORDS
    for col in ("topic", *PARALLEL_COLUMNS):
        assert col in df.columns
        assert df[col].notna().all()

    # Concurrency: check that cell tasks from different columns overlapped
    traces = result.task_traces
    assert traces, "No task traces recorded - async_trace may not be enabled"

    by_col: dict[str, list] = defaultdict(list)
    for t in traces:
        if t.task_type == "cell" and t.status == "ok" and t.slot_acquired_at and t.completed_at:
            by_col[t.column].append(t)

    overlap_found = False
    cols = [c for c in PARALLEL_COLUMNS if by_col[c]]
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            for ta in by_col[col_a]:
                for tb in by_col[col_b]:
                    if ta.slot_acquired_at < tb.completed_at and tb.slot_acquired_at < ta.completed_at:
                        overlap_found = True
                        break
                if overlap_found:
                    break
            if overlap_found:
                break
        if overlap_found:
            break

    assert overlap_found, (
        "No overlapping execution found between parallel columns - async scheduler may not be dispatching concurrently"
    )
