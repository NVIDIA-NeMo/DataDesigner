# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark DataDesigner engine performance with mock LLMs."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig, ValidationColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.mcp import MCPProvider, ToolConfig
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import SamplerType, UniformSamplerParams
from data_designer.config.validator_params import LocalCallableValidatorParams, ValidatorType
from data_designer.engine.mcp.registry import MCPToolDefinition, MCPToolResult
from data_designer.lazy_heavy_imports import np, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


RESULT_PREFIX = "BENCHMARK_RESULT="
DEFAULT_NUM_RECORDS = 1024
DEFAULT_BUFFER_SIZE = 1024
DEFAULT_SEED = 11
DEFAULT_MAX_PARALLEL_REQUESTS = 16
DEFAULT_MAX_CONCURRENT_TASKS = 64
DEFAULT_VALIDATOR_BATCH_SIZE = 256
DEFAULT_ITERATIONS = 5

MOCK_MCP_PROVIDER_NAME = "mock-mcp"
MOCK_TOOL_ALIAS = "mock-tools"
MOCK_TOOL_NAME = "mock_lookup"
MOCK_TOOL_DESCRIPTION = "Mock lookup tool for benchmark runs."
MOCK_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
}


@dataclass(frozen=True)
class BenchmarkSettings:
    num_records: int
    buffer_size: int
    seed: int
    max_parallel_requests: int
    validator_batch_size: int
    max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS
    simulated_latency: bool = False

    def to_cli_args(self) -> list[str]:
        args = [
            "--num-records",
            str(self.num_records),
            "--buffer-size",
            str(self.buffer_size),
            "--seed",
            str(self.seed),
            "--max-parallel-requests",
            str(self.max_parallel_requests),
            "--validator-batch-size",
            str(self.validator_batch_size),
            "--max-concurrent-tasks",
            str(self.max_concurrent_tasks),
        ]
        if self.simulated_latency:
            args.append("--simulated-latency")
        return args


@dataclass(frozen=True)
class BenchmarkResult:
    engine_mode: str
    num_records: int
    buffer_size: int
    build_time_sec: float
    total_time_sec: float
    dataset_hash: str
    row_count: int
    column_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_mode": self.engine_mode,
            "num_records": self.num_records,
            "buffer_size": self.buffer_size,
            "build_time_sec": self.build_time_sec,
            "total_time_sec": self.total_time_sec,
            "dataset_hash": self.dataset_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkResult:
        return cls(
            engine_mode=str(payload["engine_mode"]),
            num_records=int(payload["num_records"]),
            buffer_size=int(payload["buffer_size"]),
            build_time_sec=float(payload["build_time_sec"]),
            total_time_sec=float(payload["total_time_sec"]),
            dataset_hash=str(payload["dataset_hash"]),
            row_count=int(payload["row_count"]),
            column_count=int(payload["column_count"]),
        )


@dataclass(frozen=True)
class MetricStats:
    mean: float
    stdev: float
    ci_half_width: float
    n: int

    @property
    def ci_low(self) -> float:
        return self.mean - self.ci_half_width

    @property
    def ci_high(self) -> float:
        return self.mean + self.ci_half_width


@dataclass(frozen=True)
class ResponseProfile:
    label: str
    score_mu: float
    score_sigma: float
    latency_alpha: float
    latency_beta: float
    volatility_sigma: float
    categories: tuple[str, ...]
    category_weights: tuple[float, ...]


MODEL_PROFILES: dict[str, ResponseProfile] = {
    "mock-alpha": ResponseProfile(
        label="alpha",
        score_mu=0.1,
        score_sigma=0.35,
        latency_alpha=2.2,
        latency_beta=6.0,
        volatility_sigma=0.25,
        categories=("low", "mid", "high"),
        category_weights=(0.25, 0.55, 0.2),
    ),
    "mock-beta": ResponseProfile(
        label="beta",
        score_mu=0.3,
        score_sigma=0.45,
        latency_alpha=2.6,
        latency_beta=4.8,
        volatility_sigma=0.3,
        categories=("low", "mid", "high"),
        category_weights=(0.2, 0.5, 0.3),
    ),
    "mock-gamma": ResponseProfile(
        label="gamma",
        score_mu=0.5,
        score_sigma=0.5,
        latency_alpha=3.0,
        latency_beta=3.6,
        volatility_sigma=0.35,
        categories=("low", "mid", "high"),
        category_weights=(0.15, 0.45, 0.4),
    ),
}

DEFAULT_PROFILE = ResponseProfile(
    label="default",
    score_mu=0.2,
    score_sigma=0.4,
    latency_alpha=2.4,
    latency_beta=5.0,
    volatility_sigma=0.3,
    categories=("low", "mid", "high"),
    category_weights=(0.3, 0.5, 0.2),
)


@dataclass(frozen=True)
class FakeMessage:
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


@dataclass(frozen=True)
class FakeChoice:
    message: FakeMessage


@dataclass(frozen=True)
class FakeResponse:
    choices: list[FakeChoice]
    usage: Any | None = None
    model: str | None = None
    latency_ms: float = 0.0


def _distinct_parallel_requests(base: int) -> tuple[int, int, int]:
    if base < 3:
        raise ValueError("max_parallel_requests must be >= 3 to create distinct per-model limits.")
    high = base
    mid = max(1, int(round(high / 2)))
    low = max(1, int(round(high / 5)))

    if mid >= high:
        mid = high - 1
    if low >= mid:
        low = max(1, mid - 1)

    return high, mid, low


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(df, 1.96)


def _compute_stats(values: list[float]) -> MetricStats:
    if not values:
        return MetricStats(mean=0.0, stdev=0.0, ci_half_width=0.0, n=0)
    if len(values) == 1:
        return MetricStats(mean=values[0], stdev=0.0, ci_half_width=0.0, n=1)
    stdev = statistics.stdev(values)
    mean = statistics.mean(values)
    t_value = _t_critical_95(len(values) - 1)
    ci_half_width = t_value * stdev / math.sqrt(len(values))
    return MetricStats(mean=mean, stdev=stdev, ci_half_width=ci_half_width, n=len(values))


def _format_stats(stats: MetricStats, *, unit: str, precision: int = 3) -> str:
    fmt = f"{{:.{precision}f}}"
    mean = fmt.format(stats.mean)
    ci = fmt.format(stats.ci_half_width)
    stdev = fmt.format(stats.stdev)
    return f"{mean}{unit} ± {ci}{unit} (stdev {stdev}{unit}, n={stats.n})"


def _format_speed_stats(stats: MetricStats, *, precision: int = 2) -> str:
    fmt = f"{{:.{precision}f}}"
    mean = fmt.format(stats.mean)
    ci = fmt.format(stats.ci_half_width)
    stdev = fmt.format(stats.stdev)
    return f"{mean}x ± {ci}x (stdev {stdev}x, n={stats.n})"


def _significant_diff(stats: MetricStats) -> bool:
    return stats.n > 1 and abs(stats.mean) > stats.ci_half_width


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _stable_seed(model: str, messages: list[dict[str, Any]]) -> int:
    payload = json.dumps(
        {"model": model, "messages": messages},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _profile_for_model(model: str) -> ResponseProfile:
    for key, profile in MODEL_PROFILES.items():
        if key in model:
            return profile
    return DEFAULT_PROFILE


def _mock_response_text(model: str, messages: list[dict[str, Any]]) -> tuple[str, float]:
    profile = _profile_for_model(model)
    rng = random.Random(_stable_seed(model, messages))
    category = rng.choices(profile.categories, weights=profile.category_weights, k=1)[0]
    score = rng.lognormvariate(profile.score_mu, profile.score_sigma)
    latency_ms = int(rng.betavariate(profile.latency_alpha, profile.latency_beta) * 900.0)
    volatility = rng.gauss(0.0, profile.volatility_sigma)
    text = f"{profile.label}:{category}|score={score:.3f}|latency_ms={latency_ms}|vol={volatility:.3f}"
    return text, float(latency_ms)


def _tool_call_id(model: str, messages: list[dict[str, Any]]) -> str:
    call_seed = _stable_seed(model, messages)
    return f"tool-{call_seed:016x}"


def _tool_call_arguments(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(_stable_seed(model, messages))
    return {
        "query": f"{model}-lookup-{rng.randint(1000, 9999)}",
        "limit": rng.randint(1, 3),
    }


def _build_tool_call(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    arguments = _tool_call_arguments(model, messages)
    return {
        "id": _tool_call_id(model, messages),
        "type": "function",
        "function": {"name": MOCK_TOOL_NAME, "arguments": json.dumps(arguments)},
    }


def _should_request_tool(messages: list[dict[str, Any]]) -> bool:
    return not any(message.get("role") == "tool" for message in messages)


def _mock_tool_definition() -> MCPToolDefinition:
    return MCPToolDefinition(
        name=MOCK_TOOL_NAME,
        description=MOCK_TOOL_DESCRIPTION,
        input_schema=MOCK_TOOL_SCHEMA,
    )


def _mock_tool_result(tool_name: str, arguments: dict[str, Any], provider_name: str) -> MCPToolResult:
    payload = {
        "tool": tool_name,
        "provider": provider_name,
        "query": arguments.get("query", ""),
        "limit": arguments.get("limit", 0),
        "status": "ok",
    }
    return MCPToolResult(content=json.dumps(payload))


def _fake_response(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> FakeResponse:
    if kwargs.get("tools") and _should_request_tool(messages):
        tool_call = _build_tool_call(model, messages)
        # Compute latency for tool-call responses using the same profile/seed mechanism.
        profile = _profile_for_model(model)
        rng = random.Random(_stable_seed(model, messages))
        latency_ms = float(int(rng.betavariate(profile.latency_alpha, profile.latency_beta) * 900.0))
        return FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content="Using tool.", tool_calls=[tool_call]))],
            model=model,
            latency_ms=latency_ms,
        )
    response_text, latency_ms = _mock_response_text(model, messages)
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content=response_text))],
        model=model,
        latency_ms=latency_ms,
    )


@contextlib.contextmanager
def _patch_llm_responses(*, simulated_latency: bool = False) -> Iterator[None]:
    # Imports are deferred so engine selection respects DATA_DESIGNER_ASYNC_ENGINE.
    from data_designer.engine.models.litellm_overrides import CustomRouter

    original_completion = CustomRouter.completion
    original_acompletion = getattr(CustomRouter, "acompletion", None)

    def fake_completion(self: Any, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> FakeResponse:
        _ = self
        response = _fake_response(model, messages, **kwargs)
        if simulated_latency and response.latency_ms > 0:
            time.sleep(response.latency_ms / 1000.0)
        return response

    async def fake_acompletion(self: Any, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> FakeResponse:
        _ = self
        response = _fake_response(model, messages, **kwargs)
        if simulated_latency and response.latency_ms > 0:
            await asyncio.sleep(response.latency_ms / 1000.0)
        return response

    CustomRouter.completion = fake_completion
    CustomRouter.acompletion = fake_acompletion
    try:
        yield
    finally:
        CustomRouter.completion = original_completion
        if original_acompletion is not None:
            CustomRouter.acompletion = original_acompletion
        else:
            try:
                delattr(CustomRouter, "acompletion")
            except AttributeError:
                pass


@contextlib.contextmanager
def _patch_mcp_io() -> Iterator[None]:
    import data_designer.engine.mcp.io as mcp_io

    original_list_tools = mcp_io.list_tools
    original_call_tools = mcp_io.call_tools

    def fake_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        if getattr(provider, "name", None) != MOCK_MCP_PROVIDER_NAME:
            return original_list_tools(provider, timeout_sec=timeout_sec)
        return (_mock_tool_definition(),)

    def fake_call_tools(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        if any(getattr(call[0], "name", None) != MOCK_MCP_PROVIDER_NAME for call in calls):
            return original_call_tools(calls, timeout_sec=timeout_sec)
        return [_mock_tool_result(tool_name, arguments, provider.name) for provider, tool_name, arguments in calls]

    mcp_io.list_tools = fake_list_tools
    mcp_io.call_tools = fake_call_tools
    try:
        yield
    finally:
        mcp_io.list_tools = original_list_tools
        mcp_io.call_tools = original_call_tools


def _extract_metric(text: str, key: str) -> float | None:
    marker = f"{key}="
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = start
    while end < len(text) and (text[end].isdigit() or text[end] in {".", "-"}):
        end += 1
    try:
        return float(text[start:end])
    except ValueError:
        return None


def _validate_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    series = df["llm_stage3"].astype(str)
    scores = series.map(lambda text: _extract_metric(text, "score"))
    latencies = series.map(lambda text: _extract_metric(text, "latency_ms"))
    scores_numeric = pd.to_numeric(scores, errors="coerce")
    latency_numeric = pd.to_numeric(latencies, errors="coerce")
    is_valid = scores_numeric.between(0.0, 10.0) & latency_numeric.between(0.0, 900.0)
    return pd.DataFrame(
        {
            "is_valid": is_valid.fillna(False).astype(bool),
            "score": scores_numeric,
            "latency_ms": latency_numeric,
        }
    )


def _build_config(settings: BenchmarkSettings) -> DataDesignerConfigBuilder:
    high_parallel, mid_parallel, low_parallel = _distinct_parallel_requests(settings.max_parallel_requests)
    model_configs = [
        ModelConfig(
            alias="mock-alpha",
            model="mock-alpha",
            provider="mock-provider",
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=high_parallel),
            skip_health_check=True,
        ),
        ModelConfig(
            alias="mock-beta",
            model="mock-beta",
            provider="mock-provider",
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=low_parallel),
            skip_health_check=True,
        ),
        ModelConfig(
            alias="mock-gamma",
            model="mock-gamma",
            provider="mock-provider",
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=mid_parallel),
            skip_health_check=True,
        ),
    ]

    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.add_tool_config(
        ToolConfig(
            tool_alias=MOCK_TOOL_ALIAS,
            providers=[MOCK_MCP_PROVIDER_NAME],
            allow_tools=[MOCK_TOOL_NAME],
            max_tool_call_turns=1,
            timeout_sec=1.0,
        )
    )
    builder.add_column(
        SamplerColumnConfig(
            name="seed_value",
            sampler_type=SamplerType.UNIFORM,
            params=UniformSamplerParams(low=0.0, high=100.0, decimal_places=3),
        )
    )
    builder.add_column(
        LLMTextColumnConfig(
            name="llm_stage1",
            model_alias="mock-alpha",
            prompt="Summarize the signal for seed {{ seed_value }}.",
        )
    )
    builder.add_column(
        LLMTextColumnConfig(
            name="llm_stage2",
            model_alias="mock-beta",
            tool_alias=MOCK_TOOL_ALIAS,
            prompt="Analyze {{ llm_stage1 }} and produce a risk assessment.",
        )
    )
    builder.add_column(
        LLMTextColumnConfig(
            name="llm_stage3",
            model_alias="mock-gamma",
            prompt="Generate a recommendation from {{ llm_stage2 }} with seed {{ seed_value }}.",
        )
    )
    builder.add_column(
        ValidationColumnConfig(
            name="llm_stage3_validation",
            target_columns=["llm_stage3"],
            validator_type=ValidatorType.LOCAL_CALLABLE,
            validator_params=LocalCallableValidatorParams(validation_function=_validate_recommendation),
            batch_size=settings.validator_batch_size,
        )
    )
    return builder


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    normalized = df.reset_index(drop=True)
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    records = normalized.to_dict(orient="records")
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_single_benchmark(settings: BenchmarkSettings, engine_mode: str) -> BenchmarkResult:
    # Imports are deferred so engine selection respects DATA_DESIGNER_ASYNC_ENGINE.
    from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
    from data_designer.engine.dataset_builders.column_wise_builder import ColumnWiseDatasetBuilder
    from data_designer.engine.model_provider import resolve_model_provider_registry
    from data_designer.engine.resources.resource_provider import create_resource_provider
    from data_designer.engine.resources.seed_reader import SeedReaderRegistry
    from data_designer.engine.secret_resolver import CompositeResolver, EnvironmentResolver, PlaintextResolver

    random.seed(settings.seed)
    np.random.seed(settings.seed)

    run_config = RunConfig(
        buffer_size=settings.buffer_size,
        disable_early_shutdown=True,
        max_conversation_restarts=0,
        max_conversation_correction_steps=0,
        max_concurrent_tasks=settings.max_concurrent_tasks,
    )
    builder = _build_config(settings)

    provider = ModelProvider(
        name="mock-provider",
        endpoint="https://mock.local",
        provider_type="openai",
        api_key="mock-key",
    )
    mcp_provider = MCPProvider(
        name=MOCK_MCP_PROVIDER_NAME,
        endpoint="https://mock.local/mcp",
        api_key="mock-mcp-key",
    )
    model_provider_registry = resolve_model_provider_registry([provider], default_provider_name=provider.name)
    secret_resolver = CompositeResolver([EnvironmentResolver(), PlaintextResolver()])

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_storage = ArtifactStorage(artifact_path=temp_dir, dataset_name=f"benchmark-{engine_mode}")
        resource_provider = create_resource_provider(
            artifact_storage=artifact_storage,
            model_configs=builder.model_configs,
            secret_resolver=secret_resolver,
            model_provider_registry=model_provider_registry,
            seed_reader_registry=SeedReaderRegistry(readers=[]),
            blob_storage=None,
            seed_dataset_source=None,
            run_config=run_config,
            mcp_providers=[mcp_provider],
            tool_configs=builder.tool_configs,
        )
        dataset_builder = ColumnWiseDatasetBuilder(
            data_designer_config=builder.build(),
            resource_provider=resource_provider,
        )

        total_start = time.perf_counter()
        with _patch_llm_responses(simulated_latency=settings.simulated_latency), _patch_mcp_io():
            build_start = time.perf_counter()
            dataset_builder.build(num_records=settings.num_records)
            build_time = time.perf_counter() - build_start
            dataset = dataset_builder.artifact_storage.load_dataset_with_dropped_columns()

        dataset_hash = _dataset_fingerprint(dataset)
        total_time = time.perf_counter() - total_start

    return BenchmarkResult(
        engine_mode=engine_mode,
        num_records=settings.num_records,
        buffer_size=settings.buffer_size,
        build_time_sec=build_time,
        total_time_sec=total_time,
        dataset_hash=dataset_hash,
        row_count=int(dataset.shape[0]),
        column_count=int(dataset.shape[1]),
    )


def _run_subprocess(settings: BenchmarkSettings, engine_mode: str) -> BenchmarkResult:
    env = os.environ.copy()
    if engine_mode == "async":
        env["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
    else:
        env.pop("DATA_DESIGNER_ASYNC_ENGINE", None)

    script_path = Path(__file__).resolve()
    cmd = [sys.executable, str(script_path), "--mode", "run", "--engine", engine_mode, *settings.to_cli_args()]
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)

    if completed.returncode != 0:
        raise RuntimeError(f"Benchmark subprocess failed.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line.removeprefix(RESULT_PREFIX))
            return BenchmarkResult.from_dict(payload)

    raise RuntimeError(
        f"Benchmark subprocess did not emit a result payload.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _format_speedup(sync_time: float, async_time: float) -> str:
    if async_time <= 0:
        return "n/a"
    return f"{(sync_time / async_time):.2f}x"


def _run_with_progress(settings: BenchmarkSettings, engine_mode: str, iteration: int, total: int) -> BenchmarkResult:
    print(f"[{iteration}/{total}] Running {engine_mode} benchmark...", end="", flush=True)
    result = _run_subprocess(settings, engine_mode)
    print(f" done ({result.total_time_sec:.3f}s)")
    return result


def _compare_runs(settings: BenchmarkSettings, iterations: int) -> int:
    sync_results: list[BenchmarkResult] = []
    async_results: list[BenchmarkResult] = []
    expected_hash: str | None = None

    for iteration in range(1, iterations + 1):
        sync_result = _run_with_progress(settings, "sync", iteration, iterations)
        async_result = _run_with_progress(settings, "async", iteration, iterations)

        if sync_result.dataset_hash != async_result.dataset_hash:
            print(
                "Content mismatch detected: "
                f"sync hash {sync_result.dataset_hash} vs async hash {async_result.dataset_hash}"
            )
            return 1

        if expected_hash is None:
            expected_hash = sync_result.dataset_hash
        elif expected_hash != sync_result.dataset_hash or expected_hash != async_result.dataset_hash:
            print("Content mismatch detected across iterations.")
            return 1

        sync_results.append(sync_result)
        async_results.append(async_result)

    build_sync = [result.build_time_sec for result in sync_results]
    build_async = [result.build_time_sec for result in async_results]
    total_sync = [result.total_time_sec for result in sync_results]
    total_async = [result.total_time_sec for result in async_results]

    build_speedups = [sync / async_ for sync, async_ in zip(build_sync, build_async)]
    total_speedups = [sync / async_ for sync, async_ in zip(total_sync, total_async)]
    build_diffs = [sync - async_ for sync, async_ in zip(build_sync, build_async)]
    total_diffs = [sync - async_ for sync, async_ in zip(total_sync, total_async)]

    build_sync_stats = _compute_stats(build_sync)
    build_async_stats = _compute_stats(build_async)
    total_sync_stats = _compute_stats(total_sync)
    total_async_stats = _compute_stats(total_async)

    build_speed_stats = _compute_stats(build_speedups)
    total_speed_stats = _compute_stats(total_speedups)
    build_diff_stats = _compute_stats(build_diffs)
    total_diff_stats = _compute_stats(total_diffs)

    latency_label = "on" if settings.simulated_latency else "off"
    print("\nEngine benchmark summary (95% CI)")
    print(f"- runs: {iterations} | content match: yes | hash {expected_hash}")
    print(f"- simulated latency: {latency_label}")
    print(f"- build time sync:  {_format_stats(build_sync_stats, unit='s')}")
    print(f"- build time async: {_format_stats(build_async_stats, unit='s')}")
    print(
        f"- build speedup:    {_format_speed_stats(build_speed_stats)} | "
        f"paired diff {_format_stats(build_diff_stats, unit='s')} | "
        f"significant: {'yes' if _significant_diff(build_diff_stats) else 'no'}"
    )
    print(f"- total time sync:  {_format_stats(total_sync_stats, unit='s')}")
    print(f"- total time async: {_format_stats(total_async_stats, unit='s')}")
    print(
        f"- total speedup:    {_format_speed_stats(total_speed_stats)} | "
        f"paired diff {_format_stats(total_diff_stats, unit='s')} | "
        f"significant: {'yes' if _significant_diff(total_diff_stats) else 'no'}"
    )

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DataDesigner engine with mock LLMs and compare async execution."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("compare", "run"),
        default="compare",
        help="Run both engines in subprocesses, or run once in the current process.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=("sync", "async"),
        default="sync",
        help="Engine mode for --mode run.",
    )
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS, help="Records to generate.")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Batch buffer size.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for determinism.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of sync/async runs to include in the compare mode.",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=DEFAULT_MAX_PARALLEL_REQUESTS,
        help="Max parallel LLM requests per model.",
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_TASKS,
        help="Max in-flight tasks in the async executor (system-level backpressure).",
    )
    parser.add_argument(
        "--validator-batch-size",
        type=int,
        default=DEFAULT_VALIDATOR_BATCH_SIZE,
        help="Batch size for the local validator.",
    )
    parser.add_argument(
        "--simulated-latency",
        action="store_true",
        default=False,
        help="Simulate LLM response latency using beta-distributed delays.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = BenchmarkSettings(
        num_records=args.num_records,
        buffer_size=args.buffer_size,
        seed=args.seed,
        max_parallel_requests=args.max_parallel_requests,
        validator_batch_size=args.validator_batch_size,
        max_concurrent_tasks=args.max_concurrent_tasks,
        simulated_latency=args.simulated_latency,
    )

    if args.mode == "compare":
        sys.exit(_compare_runs(settings, args.iterations))

    if args.engine == "async":
        os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
    else:
        os.environ.pop("DATA_DESIGNER_ASYNC_ENGINE", None)

    print(f"Running {args.engine} benchmark...")
    result = _run_single_benchmark(settings, args.engine)
    print(f"{RESULT_PREFIX}{json.dumps(result.to_dict(), sort_keys=True)}")


if __name__ == "__main__":
    main()
