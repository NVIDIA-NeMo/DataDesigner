# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: sync vs async builder with repeated trials.

Runs each engine N times (default 5), drops the first run of each as warmup,
and reports min / median / mean / max / stdev for the remaining runs.

Uses real LLMs (openai-text). Set OPENAI_API_KEY in environment.

Usage:
    cd async_test && uv run python bench_sync_vs_async.py
    cd async_test && uv run python bench_sync_vs_async.py --trials 7 --num-records 10
"""

from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
import time
import warnings
from argparse import ArgumentParser

os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"

warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*Unclosed.*")

import data_designer.engine.dataset_builders.column_wise_builder as cwb
from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.interface import DataDesigner

NUM_RECORDS_DEFAULT = 5
TRIALS_DEFAULT = 5
MAX_PARALLEL_DEFAULT = 4

# DAG shapes:
#   narrow:  topic → summary → followup              (sequential, 2 LLM cols)
#   wide:    topic → summary ─┐
#            topic → analysis  ├→ synthesis            (3 parallel + 1 merge, 4 LLM cols)
#            topic → trivia  ──┘
DAG_DEFAULT = "wide"


def _build_config(dag: str, max_parallel: int) -> DataDesignerConfigBuilder:
    config = DataDesignerConfigBuilder(
        model_configs=[
            ModelConfig(
                alias="openai-text",
                model="gpt-4.1",
                provider="openai",
                inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=max_parallel),
            )
        ]
    )
    config.add_column(
        SamplerColumnConfig(
            name="topic",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["science", "history", "art"]),
        )
    )

    if dag == "narrow":
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
    elif dag == "wide":
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
    else:
        raise ValueError(f"Unknown dag shape: {dag!r}. Use 'narrow' or 'wide'.")

    return config


def _suppress_logs() -> None:
    """Mute all data_designer / litellm / httpx loggers."""
    for name in ("data_designer", "LiteLLM", "litellm", "httpx", "asyncio"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    # Also prevent the root logger from propagating data_designer messages.
    logging.getLogger().setLevel(logging.CRITICAL)


def _run_once(*, async_mode: bool, num_records: int, dag: str, max_parallel: int) -> float:
    """Run a single creation and return wall-clock seconds."""
    _suppress_logs()
    cwb.DATA_DESIGNER_ASYNC_ENGINE = async_mode

    with tempfile.TemporaryDirectory() as artifact_dir:
        dd = DataDesigner(artifact_path=artifact_dir)
        dd.set_run_config(RunConfig(buffer_size=1000, disable_early_shutdown=True, async_trace=False))
        config = _build_config(dag, max_parallel)

        t0 = time.perf_counter()
        result = dd.create(config, num_records=num_records, dataset_name="bench")
        elapsed = time.perf_counter() - t0

        df = result.load_dataset()
        assert len(df) == num_records, f"Expected {num_records} rows, got {len(df)}"

    return elapsed


def _stats(times: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of times."""
    n = len(times)
    s = sorted(times)
    mean = sum(s) / n
    if n % 2 == 1:
        median = s[n // 2]
    else:
        median = (s[n // 2 - 1] + s[n // 2]) / 2
    variance = sum((t - mean) ** 2 for t in s) / n
    return {
        "min": s[0],
        "median": median,
        "mean": mean,
        "max": s[-1],
        "stdev": math.sqrt(variance),
    }


def _print_row(label: str, st: dict[str, float]) -> None:
    print(
        f"  {label:<7} "
        f"{st['min']:>7.3f}s  "
        f"{st['median']:>7.3f}s  "
        f"{st['mean']:>7.3f}s  "
        f"{st['max']:>7.3f}s  "
        f"{st['stdev']:>7.3f}s"
    )


def main() -> None:
    parser = ArgumentParser(description="Benchmark sync vs async builder")
    parser.add_argument(
        "--trials", type=int, default=TRIALS_DEFAULT, help="Total trials per engine (including 1 warmup)"
    )
    parser.add_argument("--num-records", type=int, default=NUM_RECORDS_DEFAULT, help="Records per trial")
    parser.add_argument(
        "--dag",
        type=str,
        default=DAG_DEFAULT,
        choices=["narrow", "wide"],
        help="DAG shape: narrow (sequential) or wide (parallel branches)",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=MAX_PARALLEL_DEFAULT, help="max_parallel_requests for the LLM model"
    )
    args = parser.parse_args()

    trials: int = args.trials
    num_records: int = args.num_records
    dag: str = args.dag
    max_parallel: int = args.max_parallel

    if trials < 2:
        print("Need at least 2 trials (1 warmup + 1 measured). Setting trials=2.")
        trials = 2

    measured = trials - 1

    print("=" * 70)
    print(f"Benchmark: Sync vs Async  ({trials} trials, first dropped as warmup)")
    print(f"Records per trial: {num_records}, DAG: {dag}, max_parallel: {max_parallel}")
    print("=" * 70)

    # --- Warmup: one sync run to prime health checks, caches, etc. ---
    # First DataDesigner() call configures logging, so suppress via devnull.
    print("\nWarmup (sync)...", end=" ", flush=True)
    devnull = open(os.devnull, "w")  # noqa: SIM115
    old_stderr = sys.stderr
    sys.stderr = devnull
    warmup_time = _run_once(async_mode=False, num_records=num_records, dag=dag, max_parallel=max_parallel)
    sys.stderr = old_stderr
    devnull.close()
    print(f"{warmup_time:.3f}s")

    # --- Interleaved trials: ABABABAB to reduce temporal bias ---
    sync_times: list[float] = []
    async_times: list[float] = []

    for i in range(measured):
        trial_num = i + 1

        # Sync
        print(f"\n  Trial {trial_num}/{measured} — sync ...", end=" ", flush=True)
        t_sync = _run_once(async_mode=False, num_records=num_records, dag=dag, max_parallel=max_parallel)
        sync_times.append(t_sync)
        print(f"{t_sync:.3f}s")

        # Async
        print(f"  Trial {trial_num}/{measured} — async...", end=" ", flush=True)
        t_async = _run_once(async_mode=True, num_records=num_records, dag=dag, max_parallel=max_parallel)
        async_times.append(t_async)
        print(f"{t_async:.3f}s")

    # --- Stats ---
    sync_st = _stats(sync_times)
    async_st = _stats(async_times)
    speedup_median = sync_st["median"] / async_st["median"] if async_st["median"] > 0 else float("inf")
    speedup_mean = sync_st["mean"] / async_st["mean"] if async_st["mean"] > 0 else float("inf")

    print("\n" + "=" * 70)
    print(f"Results ({measured} measured trials, {num_records} records each, dag={dag})")
    print("=" * 70)

    header = f"  {'engine':<7} {'min':>8}  {'median':>8}  {'mean':>8}  {'max':>8}  {'stdev':>8}"
    print(header)
    print(f"  {'-' * (len(header.strip()) - 2)}")
    _print_row("sync", sync_st)
    _print_row("async", async_st)

    print(f"\n  Speedup (median): {speedup_median:.2f}x")
    print(f"  Speedup (mean):   {speedup_mean:.2f}x")

    # Per-trial detail
    print("\n  Per-trial times:")
    print(f"  {'trial':>5}  {'sync':>8}  {'async':>8}  {'ratio':>7}")
    print(f"  {'-' * 33}")
    for i, (s, a) in enumerate(zip(sync_times, async_times)):
        ratio = s / a if a > 0 else float("inf")
        print(f"  {i + 1:>5}  {s:>7.3f}s  {a:>7.3f}s  {ratio:>6.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
