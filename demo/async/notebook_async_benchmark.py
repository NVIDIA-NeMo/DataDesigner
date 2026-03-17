# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %% [markdown]
# # Async Scheduler Benchmark
#
# This notebook compares the **sync** (column-by-column) builder with the
# **async** (task-queue) scheduler and shows when the async path delivers
# real speedups.
#
# **Key finding:** the async scheduler needs *both* a wide DAG (independent
# columns) *and* enough LLM concurrency (`max_parallel_requests`) to
# outperform the sync engine. With a narrow/sequential DAG or low
# concurrency, both engines hit the same API throughput ceiling.
#
# **Requirements:** `OPENAI_API_KEY` in environment. Uses `gpt-4.1` via
# `openai-text` alias.

# %% Setup
from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
import time
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"

warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*Unclosed.*")

from IPython.display import HTML, display

import data_designer.engine.dataset_builders.column_wise_builder as cwb
from data_designer.config.column_configs import (
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.interface import DataDesigner

NUM_RECORDS = 10
TRIALS = 5

_STRATEGY_MAP: dict[type, GenerationStrategy] = {
    SamplerColumnConfig: GenerationStrategy.FULL_COLUMN,
    LLMTextColumnConfig: GenerationStrategy.CELL_BY_CELL,
}


def suppress_logs() -> None:
    for name in ("data_designer", "LiteLLM", "litellm", "httpx", "asyncio"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)


def render_mermaid(code: str) -> None:
    """Render a Mermaid diagram inline via the Mermaid JS CDN."""
    display(
        HTML(
            f"""<pre class="mermaid">{code}</pre>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true }});
</script>"""
        )
    )


def _execution_graph_from_builder(config: DataDesignerConfigBuilder):  # noqa: ANN201
    """Build an ExecutionGraph directly from a config builder."""
    cols = list(config._column_configs.values())
    strategies = {c.name: _STRATEGY_MAP[type(c)] for c in cols}
    return ExecutionGraph.create(cols, strategies)


# %% [markdown]
# ## DAG Shapes
#
# We test two DAG shapes:
#
# **Narrow** (sequential): `topic → summary → followup`
# - 2 LLM columns, fully sequential — no cross-column parallelism possible.
#
# **Wide** (parallel branches):
# ```
# topic → summary  ─┐
# topic → analysis  ─┼→ synthesis
# topic → trivia   ─┘
# ```
# - 3 independent LLM columns off the sampler, then 1 merge column.
# - The async scheduler can overlap summary/analysis/trivia.

# %% Config builders


def _model_configs(max_parallel: int) -> list[ModelConfig]:
    return [
        ModelConfig(
            alias="openai-text",
            model="gpt-4.1",
            provider="openai",
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=max_parallel),
        )
    ]


def build_narrow(max_parallel: int) -> DataDesignerConfigBuilder:
    config = DataDesignerConfigBuilder(model_configs=_model_configs(max_parallel))
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


def build_wide(max_parallel: int) -> DataDesignerConfigBuilder:
    config = DataDesignerConfigBuilder(model_configs=_model_configs(max_parallel))
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
            name="analysis",
            model_alias="openai-text",
            prompt="Write a one-sentence analysis of why {{ topic }} matters.",
        )
    )
    config.add_column(
        LLMTextColumnConfig(
            name="trivia",
            model_alias="openai-text",
            prompt="State one surprising fact about {{ topic }}.",
        )
    )
    config.add_column(
        LLMTextColumnConfig(
            name="synthesis",
            model_alias="openai-text",
            prompt=(
                "Combine these into one sentence: "
                "summary='{{ summary }}', analysis='{{ analysis }}', trivia='{{ trivia }}'."
            ),
        )
    )
    return config


# %% [markdown]
# ## DAG Visualization
#
# The `ExecutionGraph` class models the column dependency DAG and annotates
# each node with its generation strategy (`full_column` for samplers,
# `cell_by_cell` for LLM columns). The `to_mermaid()` method renders this
# as a Mermaid flowchart.

# %%
narrow_graph = _execution_graph_from_builder(build_narrow(4))
wide_graph = _execution_graph_from_builder(build_wide(4))

print("Narrow DAG")
print(f"  Critical path: {' → '.join(narrow_graph.get_longest_dependency_chain())}")
print(f"  Task count (10 records, buffer=1000): {narrow_graph.compute_task_count(NUM_RECORDS, 1000)}")
render_mermaid(narrow_graph.to_mermaid())

# %%
print("Wide DAG")
print(f"  Critical path: {' → '.join(wide_graph.get_longest_dependency_chain())}")
print(f"  Task count (10 records, buffer=1000): {wide_graph.compute_task_count(NUM_RECORDS, 1000)}")
render_mermaid(wide_graph.to_mermaid())


# %% Runner


def run_once(
    *,
    async_mode: bool,
    config: DataDesignerConfigBuilder,
    trace: bool = False,
) -> tuple[float, list]:
    """Run one creation. Returns (elapsed_seconds, traces)."""
    suppress_logs()
    cwb.DATA_DESIGNER_ASYNC_ENGINE = async_mode

    with tempfile.TemporaryDirectory() as artifact_dir:
        dd = DataDesigner(artifact_path=artifact_dir)
        dd.set_run_config(
            RunConfig(
                buffer_size=1000,
                disable_early_shutdown=True,
                async_trace=trace and async_mode,
            )
        )

        t0 = time.perf_counter()
        result = dd.create(config, num_records=NUM_RECORDS, dataset_name="bench")
        elapsed = time.perf_counter() - t0

        df = result.load_dataset()
        assert len(df) == NUM_RECORDS, f"Expected {NUM_RECORDS} rows, got {len(df)}"

        traces = result.task_traces if hasattr(result, "task_traces") else []

    return elapsed, traces


def benchmark(
    config_fn,
    max_parallel: int,
    label: str,
    trace_last: bool = False,
) -> dict:
    """Run TRIALS interleaved sync/async and return stats + optional traces."""
    config_s = config_fn(max_parallel)
    config_a = config_fn(max_parallel)

    # Warmup
    devnull = open(os.devnull, "w")  # noqa: SIM115
    old_stderr = sys.stderr
    sys.stderr = devnull
    run_once(async_mode=False, config=config_s)
    sys.stderr = old_stderr
    devnull.close()

    sync_times: list[float] = []
    async_times: list[float] = []
    last_traces: list = []

    for i in range(TRIALS):
        is_last = i == TRIALS - 1

        config_s = config_fn(max_parallel)
        t_sync, _ = run_once(async_mode=False, config=config_s)
        sync_times.append(t_sync)

        config_a = config_fn(max_parallel)
        t_async, traces = run_once(
            async_mode=True,
            config=config_a,
            trace=trace_last and is_last,
        )
        async_times.append(t_async)
        if is_last:
            last_traces = traces

    return {
        "label": label,
        "max_parallel": max_parallel,
        "sync": stats(sync_times),
        "async": stats(async_times),
        "sync_times": sync_times,
        "async_times": async_times,
        "traces": last_traces,
    }


def stats(times: list[float]) -> dict[str, float]:
    n = len(times)
    s = sorted(times)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    variance = sum((t - mean) ** 2 for t in s) / n
    return {"min": s[0], "median": median, "mean": mean, "max": s[-1], "stdev": math.sqrt(variance)}


# %% [markdown]
# ## Experiment 1: Narrow DAG, default concurrency (4)
#
# With a sequential DAG and low concurrency, the async scheduler has no room
# to parallelize. Both engines process one column at a time with 4 workers.

# %%
print("Running: narrow DAG, max_parallel=4 ...")
r_narrow_4 = benchmark(build_narrow, max_parallel=4, label="narrow, par=4")

# %% [markdown]
# ## Experiment 2: Wide DAG, default concurrency (4)
#
# Even with 3 independent branches, `max_parallel_requests=4` means only 4
# LLM calls can be in flight at once. The sync engine already saturates
# these 4 slots column-by-column. The async scheduler's cross-column
# dispatch hits the same ceiling.

# %%
print("Running: wide DAG, max_parallel=4 ...")
r_wide_4 = benchmark(build_wide, max_parallel=4, label="wide, par=4")

# %% [markdown]
# ## Experiment 3: Wide DAG, high concurrency (16)
#
# Now we raise `max_parallel_requests=16`. The sync engine still processes
# columns one at a time (16 workers per column). The async scheduler can
# dispatch tasks from summary, analysis, and trivia *simultaneously*,
# keeping 16 slots busy across columns.

# %%
print("Running: wide DAG, max_parallel=16 ...")
r_wide_16 = benchmark(build_wide, max_parallel=16, label="wide, par=16", trace_last=True)

# %% [markdown]
# ## Results Summary

# %%
rows = []
for r in [r_narrow_4, r_wide_4, r_wide_16]:
    s, a = r["sync"], r["async"]
    speedup = s["median"] / a["median"] if a["median"] > 0 else float("inf")
    rows.append(
        {
            "Configuration": r["label"],
            "Sync median (s)": round(s["median"], 1),
            "Async median (s)": round(a["median"], 1),
            "Speedup": f"{speedup:.2f}x",
        }
    )

summary_df = pd.DataFrame(rows)
display(summary_df.style.hide(axis="index").set_caption(f"Benchmark results ({TRIALS} trials, {NUM_RECORDS} records)"))

# Per-trial detail
for r in [r_narrow_4, r_wide_4, r_wide_16]:
    trial_rows = []
    for i, (st, at) in enumerate(zip(r["sync_times"], r["async_times"])):
        ratio = st / at if at > 0 else float("inf")
        trial_rows.append(
            {"Trial": i + 1, "Sync (s)": round(st, 1), "Async (s)": round(at, 1), "Ratio": f"{ratio:.2f}x"}
        )
    trial_df = pd.DataFrame(trial_rows)
    display(trial_df.style.hide(axis="index").set_caption(r["label"]))

# %% [markdown]
# ## Using Traces to Optimize
#
# The async scheduler records a `TaskTrace` for every dispatched task when
# `async_trace=True`. Each trace captures:
#
# | Field | Meaning |
# |---|---|
# | `column` | Which column this task generates |
# | `row_group` / `row_index` | Position in the dataset |
# | `task_type` | `from_scratch`, `batch`, or `cell` |
# | `dispatched_at` | When the scheduler queued the task |
# | `slot_acquired_at` | When it started executing (after waiting for a slot) |
# | `completed_at` | When it finished |
# | `status` | `ok` or `error` |
#
# From these we can compute **wait time** (queued → slot acquired) and
# **run time** (slot acquired → completed) for every task, revealing where
# the pipeline spends its time.

# %%

traces = r_wide_16["traces"]

if traces:
    t0 = min(t.dispatched_at for t in traces if t.dispatched_at)

    by_column: dict[str, list] = {}
    for t in traces:
        by_column.setdefault(t.column, []).append(t)

    # --- Per-column summary ---
    col_rows = []
    for col in sorted(by_column):
        col_traces = by_column[col]
        waits = [
            (t.slot_acquired_at - t.dispatched_at) * 1000 for t in col_traces if t.slot_acquired_at and t.dispatched_at
        ]
        runs = [
            (t.completed_at - t.slot_acquired_at) * 1000 for t in col_traces if t.completed_at and t.slot_acquired_at
        ]
        errors = sum(1 for t in col_traces if t.status == "error")
        col_rows.append(
            {
                "Column": col,
                "Tasks": len(col_traces),
                "Avg Wait (ms)": round(sum(waits) / len(waits), 0) if waits else 0,
                "Avg Run (ms)": round(sum(runs) / len(runs), 0) if runs else 0,
                "Total Run (ms)": round(sum(runs), 0),
                "Errors": errors,
            }
        )

    col_df = pd.DataFrame(col_rows)
    display(
        col_df.style.hide(axis="index")
        .set_caption("Per-column timing (wide DAG, max_parallel=16, async)")
        .background_gradient(subset=["Total Run (ms)"], cmap="YlOrRd")
    )

    # --- Column timeline ---
    timeline_rows = []
    for col in sorted(by_column):
        col_traces = by_column[col]
        first = min(t.dispatched_at for t in col_traces if t.dispatched_at) - t0
        last = max(t.completed_at for t in col_traces if t.completed_at) - t0
        timeline_rows.append(
            {
                "Column": col,
                "First Dispatch (s)": round(first, 3),
                "Last Complete (s)": round(last, 3),
                "Span (s)": round(last - first, 3),
            }
        )

    timeline_df = pd.DataFrame(timeline_rows)
    display(
        timeline_df.style.hide(axis="index")
        .set_caption("Column timeline (relative to first dispatch)")
        .background_gradient(subset=["Span (s)"], cmap="Blues")
    )

# %% [markdown]
# ### Per-row-group breakdown
#
# The async scheduler pipelines row groups: it can start independent columns
# for row group 1 while row group 0 is still generating dependent columns
# like `synthesis`. This table shows when each (row group, column) pair
# starts and finishes.

# %%
if traces:
    rg_rows = []
    for t in traces:
        if t.dispatched_at and t.completed_at:
            rg_rows.append(
                {
                    "Row Group": t.row_group,
                    "Column": t.column,
                    "Task Type": t.task_type,
                    "Row Index": t.row_index if t.row_index is not None else "—",
                    "Start (s)": round(t.slot_acquired_at - t0, 3) if t.slot_acquired_at else None,
                    "End (s)": round(t.completed_at - t0, 3),
                    "Run (ms)": (
                        round((t.completed_at - t.slot_acquired_at) * 1000, 0) if t.slot_acquired_at else None
                    ),
                    "Status": t.status,
                }
            )

    rg_df = pd.DataFrame(rg_rows).sort_values(["Row Group", "Start (s)"], na_position="last")

    # Summary per (row_group, column)
    rg_summary = (
        rg_df.groupby(["Row Group", "Column"])
        .agg(
            Tasks=("Run (ms)", "count"),
            First_Start=("Start (s)", "min"),
            Last_End=("End (s)", "max"),
            Avg_Run_ms=("Run (ms)", "mean"),
        )
        .reset_index()
    )
    rg_summary["Avg_Run_ms"] = rg_summary["Avg_Run_ms"].round(0)
    rg_summary.columns = ["Row Group", "Column", "Tasks", "First Start (s)", "Last End (s)", "Avg Run (ms)"]

    display(
        rg_summary.style.hide(axis="index")
        .set_caption("Per-row-group column breakdown")
        .background_gradient(subset=["Avg Run (ms)"], cmap="YlOrRd")
    )
else:
    print("(no traces captured — run with trace=True)")

# %% [markdown]
# ### Task Gantt Chart
#
# Each bar represents a task executing on the async scheduler. Bars are
# color-coded by column — overlapping bars across columns show the async
# scheduler exploiting cross-column parallelism.
#
# Notice that `synthesis` bars start *before* `summary`/`analysis`/`trivia`
# have finished all their rows. This is because `synthesis` is a
# **cell-by-cell** column: it only needs `summary[i]`, `analysis[i]`, and
# `trivia[i]` to be complete before generating `synthesis[i]`. The scheduler
# dispatches each synthesis cell as soon as its per-row dependencies are met,
# without waiting for the full column to finish.

# %%
if traces:
    end_time = max(t.completed_at for t in traces if t.completed_at)
    total_secs = end_time - t0

    # Use topological order so the chart reads top-to-bottom like the DAG
    columns_ordered = wide_graph.get_topological_order()
    cmap = plt.get_cmap("tab10")
    col_colors = {col: cmap(i) for i, col in enumerate(columns_ordered)}

    # Build bars: one per task, y-position = column index
    fig, ax = plt.subplots(figsize=(12, max(3, len(columns_ordered) * 0.8)))

    for col_idx, col in enumerate(columns_ordered):
        for t in by_column[col]:
            if t.slot_acquired_at and t.completed_at:
                start = t.slot_acquired_at - t0
                duration = t.completed_at - t.slot_acquired_at
                alpha = 0.4 if t.status == "error" else 0.85
                ax.barh(
                    col_idx,
                    duration,
                    left=start,
                    height=0.6,
                    color=col_colors[col],
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.5,
                )

    ax.set_yticks(range(len(columns_ordered)))
    ax.set_yticklabels(columns_ordered)
    ax.set_xlabel("Time (s)")
    ax.set_title("Task Gantt Chart — wide DAG, max_parallel=16, async")
    ax.set_xlim(0, total_secs * 1.02)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=col_colors[c], label=c) for c in columns_ordered]
    ax.legend(handles=patches, loc="upper right", fontsize="small", framealpha=0.8)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Cell-level Timeline
#
# This chart shows every individual cell task, with the y-axis representing
# row indices. Each colored bar is one (column, row) task. You can see how
# `synthesis` cells start filling in from the top (row 0) while the
# independent columns are still processing later rows — this is the
# cell-level pipelining that the async scheduler enables.

# %%
if traces:
    # Collect cell-by-cell tasks (row_index is not None)
    cell_tasks = [t for t in traces if t.row_index is not None and t.slot_acquired_at and t.completed_at]
    # Also include full_column tasks, drawn as spanning all rows
    batch_tasks = [t for t in traces if t.row_index is None and t.slot_acquired_at and t.completed_at]

    if cell_tasks:
        row_indices = sorted({t.row_index for t in cell_tasks})
        columns_ordered = wide_graph.get_topological_order()
        cmap_cell = plt.get_cmap("tab10")
        col_colors = {col: cmap_cell(i) for i, col in enumerate(columns_ordered)}
        n_cols = len(columns_ordered)

        bar_height = 0.8 / n_cols  # subdivide each row's vertical space

        fig, ax = plt.subplots(figsize=(12, max(3, len(row_indices) * 0.6)))

        for t in cell_tasks:
            col_offset = columns_ordered.index(t.column)
            y = t.row_index + (col_offset - n_cols / 2 + 0.5) * bar_height
            start = t.slot_acquired_at - t0
            duration = t.completed_at - t.slot_acquired_at
            alpha = 0.4 if t.status == "error" else 0.85
            ax.barh(
                y,
                duration,
                left=start,
                height=bar_height * 0.9,
                color=col_colors[t.column],
                alpha=alpha,
                edgecolor="white",
                linewidth=0.3,
            )

        # Show full_column tasks as a thin bar spanning all rows
        for t in batch_tasks:
            start = t.slot_acquired_at - t0
            duration = t.completed_at - t.slot_acquired_at
            ax.barh(
                len(row_indices) / 2 - 0.5,
                duration,
                left=start,
                height=len(row_indices) * 0.05,
                color=col_colors.get(t.column, "gray"),
                alpha=0.3,
                edgecolor="none",
            )

        ax.set_yticks(row_indices)
        ax.set_yticklabels([f"row {r}" for r in row_indices])
        ax.set_xlabel("Time (s)")
        ax.set_title("Cell-level Timeline — each bar is one (column, row) task")
        ax.set_xlim(0, total_secs * 1.02)
        ax.invert_yaxis()

        patches = [mpatches.Patch(color=col_colors[c], label=c) for c in columns_ordered]
        ax.legend(handles=patches, loc="upper right", fontsize="small", framealpha=0.8)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ### Optimization Hints

# %%
if traces:
    print("Optimization hints:")

    # 1. Slowest column by total run time
    max_col = max(
        by_column,
        key=lambda c: sum(
            (t.completed_at - t.slot_acquired_at) for t in by_column[c] if t.completed_at and t.slot_acquired_at
        ),
    )
    max_total = sum(
        (t.completed_at - t.slot_acquired_at) for t in by_column[max_col] if t.completed_at and t.slot_acquired_at
    )
    print(f"  - Slowest column: '{max_col}' ({max_total:.1f}s total LLM time)")

    # 2. Average wait time
    all_waits = [(t.slot_acquired_at - t.dispatched_at) for t in traces if t.slot_acquired_at and t.dispatched_at]
    avg_wait = sum(all_waits) / len(all_waits) if all_waits else 0
    if avg_wait > 0.5:
        print(f"  - High avg wait: {avg_wait:.3f}s → consider increasing max_parallel_requests")
    else:
        print(f"  - Low avg wait: {avg_wait:.3f}s → concurrency slots are well-utilized")

    # 3. Column overlap detection
    col_spans = {}
    for col in by_column:
        first = min(t.dispatched_at for t in by_column[col] if t.dispatched_at)
        last = max(t.completed_at for t in by_column[col] if t.completed_at)
        col_spans[col] = (first, last)

    overlapping_pairs = []
    cols = sorted(col_spans)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            s1, e1 = col_spans[c1]
            s2, e2 = col_spans[c2]
            overlap = max(0, min(e1, e2) - max(s1, s2))
            if overlap > 0:
                overlapping_pairs.append((c1, c2, overlap))

    if overlapping_pairs:
        print(f"  - Column overlap detected ({len(overlapping_pairs)} pairs):")
        for c1, c2, overlap in sorted(overlapping_pairs, key=lambda x: -x[2]):
            print(f"      {c1} ↔ {c2}: {overlap:.1f}s overlap")
    else:
        print("  - No column overlap — async scheduler isn't adding parallelism")
        print("    → check that columns have independent dependencies")
else:
    print("(no traces captured — run with trace=True)")

# %% [markdown]
# ## Takeaways
#
# 1. **DAG shape matters.** Sequential pipelines (A → B → C) give the async
#    scheduler nothing to overlap. Design pipelines with independent branches
#    where possible.
#
# 2. **`max_parallel_requests` is the lever.** The default of 4 is
#    conservative. If your provider supports higher concurrency, increase it
#    to let the scheduler fill slots across columns.
#
# 3. **Traces reveal bottlenecks.** Use `RunConfig(async_trace=True)` and
#    inspect `result.task_traces` to see:
#    - Which columns are slowest (optimize prompts or use faster models)
#    - Whether tasks are waiting for slots (increase `max_parallel_requests`)
#    - Whether columns overlap in time (confirms async is helping)
#
# 4. **No regression.** Even when the async scheduler can't parallelize,
#    it matches sync performance — the overhead is negligible.
