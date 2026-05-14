# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NUM_RECORDS = 2
PARALLEL_COLUMNS = ("summary", "analysis")


def _run_async_engine_concurrency_case(tmp_path: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    script = f"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import data_designer.config as dd
from data_designer.config.run_config import RunConfig
from data_designer.interface import DataDesigner

NUM_RECORDS = {NUM_RECORDS}
PARALLEL_COLUMNS = {PARALLEL_COLUMNS!r}
tmp_path = Path({str(tmp_path)!r})

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
            prompt="Write one sentence about {{{{ topic }}}} (" + col + ").",
        )
    )

result = dd_instance.create(config, num_records=NUM_RECORDS, dataset_name="async_e2e")
df = result.load_dataset()
traces = result.task_traces

by_col: dict[str, list[tuple[float, float]]] = defaultdict(list)
for trace in traces:
    if trace.task_type == "cell" and trace.status == "ok" and trace.slot_acquired_at and trace.completed_at:
        by_col[trace.column].append((trace.slot_acquired_at, trace.completed_at))

overlap_found = False
cols = [col for col in PARALLEL_COLUMNS if by_col[col]]
for i, col_a in enumerate(cols):
    for col_b in cols[i + 1 :]:
        for start_a, end_a in by_col[col_a]:
            for start_b, end_b in by_col[col_b]:
                if start_a < end_b and start_b < end_a:
                    overlap_found = True
                    break
            if overlap_found:
                break
        if overlap_found:
            break
    if overlap_found:
        break

payload = {{
    "rows": len(df),
    "columns": list(df.columns),
    "non_null": {{col: bool(df[col].notna().all()) for col in ("topic", *PARALLEL_COLUMNS)}},
    "trace_count": len(traces),
    "overlap_found": overlap_found,
}}
print("RESULT_JSON=" + json.dumps(payload))
"""
    env = os.environ.copy()
    env["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise AssertionError(f"Missing RESULT_JSON marker in subprocess output:\n{completed.stdout}")


def test_async_engine_concurrent_columns(tmp_path: Path) -> None:
    """Verify the async engine runs independent LLM columns concurrently."""
    if os.environ.get("NVIDIA_API_KEY") is None:
        pytest.skip("NVIDIA_API_KEY must be set")

    payload = _run_async_engine_concurrency_case(tmp_path)

    assert payload["rows"] == NUM_RECORDS
    for col in ("topic", *PARALLEL_COLUMNS):
        assert col in payload["columns"]
        assert payload["non_null"][col]

    assert payload["trace_count"] > 0
    assert payload["overlap_found"], (
        "No overlapping execution found between parallel columns - async scheduler may not be dispatching concurrently"
    )
