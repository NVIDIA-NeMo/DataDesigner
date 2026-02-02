# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark MCPIOService overhead with mocked MCP servers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import random
import time
import tracemalloc
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from data_designer.config.mcp import MCPProvider
from data_designer.engine.mcp.io import MCPIOService

try:
    import resource
except ImportError:  # pragma: no cover - resource not available on all platforms
    resource = None


BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class LatencyProfile:
    """Latency profile for mocked MCP responses."""

    name: str
    fast_range_sec: tuple[float, float]
    tail_probability: float
    tail_range_sec: tuple[float, float]
    list_tools_delay_sec: float

    def sample_delay(self, rng: random.Random) -> float:
        if self.tail_probability > 0 and rng.random() < self.tail_probability:
            return rng.uniform(*self.tail_range_sec)
        return rng.uniform(*self.fast_range_sec)


@dataclass(frozen=True)
class ProviderSpec:
    provider: MCPProvider
    tools: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FakeTransport:
    provider_name: str
    tools: tuple[dict[str, Any], ...]
    latency_profile: LatencyProfile
    rng: random.Random


@dataclass(frozen=True)
class FakeToolResult:
    content: dict[str, Any]
    isError: bool = False


@dataclass(frozen=True)
class CallSpec:
    provider: MCPProvider
    tool_name: str
    arguments: dict[str, Any]
    delay_sec: float


@dataclass(frozen=True)
class BenchmarkResult:
    scenario_name: str
    total_calls: int
    elapsed_sec: float
    rpm: float
    server_walltime_sec: float
    overhead_sec: float
    overhead_pct: float
    rss_start_mb: float | None
    rss_end_mb: float | None
    rss_growth_mb: float | None
    py_current_mb: float
    py_peak_mb: float
    py_growth_mb: float


class FakeServerContext:
    def __init__(self, transport: FakeTransport) -> None:
        self._transport = transport

    async def __aenter__(self) -> tuple[FakeTransport, None]:
        return self._transport, None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        _ = exc_type, exc, tb
        return None


class FakeClientSession:
    def __init__(self, read: FakeTransport, write: Any) -> None:
        self._transport = read
        self._write = write
        self._closed = False

    async def __aenter__(self) -> FakeClientSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        _ = exc_type, exc, tb
        self._closed = True

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        delay = self._transport.latency_profile.list_tools_delay_sec
        if delay > 0:
            await asyncio.sleep(delay)
        return list(self._transport.tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> FakeToolResult:
        delay = arguments.get("_benchmark_delay_sec")
        if delay is None:
            delay = self._transport.latency_profile.sample_delay(self._transport.rng)
        if delay > 0:
            await asyncio.sleep(delay)
        payload = {"tool": tool_name, "provider": self._transport.provider_name}
        return FakeToolResult(content=payload, isError=False)


class FakeServerRegistry:
    def __init__(self, providers: list[ProviderSpec], latency_profile: LatencyProfile, rng: random.Random) -> None:
        self._providers_by_endpoint = {spec.provider.endpoint: spec for spec in providers}
        self._latency_profile = latency_profile
        self._rng = rng

    def context_for_endpoint(self, endpoint: str) -> FakeServerContext:
        spec = self._providers_by_endpoint.get(endpoint)
        if spec is None:
            raise ValueError(f"Unknown endpoint {endpoint!r} in fake registry.")
        transport = FakeTransport(
            provider_name=spec.provider.name,
            tools=spec.tools,
            latency_profile=self._latency_profile,
            rng=self._rng,
        )
        return FakeServerContext(transport)


def _format_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} MB"


def _get_rss_bytes() -> int | None:
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss < 10_000_000:
        rss *= 1024
    return int(rss)


def _build_tools(provider_name: str, tools_per_provider: int) -> tuple[dict[str, Any], ...]:
    tools: list[dict[str, Any]] = []
    for idx in range(tools_per_provider):
        tool_name = f"{provider_name}-tool-{idx}"
        tools.append(
            {
                "name": tool_name,
                "description": f"Tool {idx} for {provider_name}.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"],
                },
            }
        )
    return tuple(tools)


def _build_provider_specs(
    provider_count: int,
    tools_per_provider: int,
) -> tuple[list[ProviderSpec], dict[str, tuple[str, ...]]]:
    providers: list[ProviderSpec] = []
    tool_names: dict[str, tuple[str, ...]] = {}
    for idx in range(provider_count):
        name = f"provider-{idx}"
        provider = MCPProvider(name=name, endpoint=f"fake://{name}", api_key="benchmark-key")
        tools = _build_tools(name, tools_per_provider)
        tool_names[name] = tuple(tool["name"] for tool in tools)
        providers.append(ProviderSpec(provider=provider, tools=tools))
    return providers, tool_names


def _build_call_specs(
    providers: list[ProviderSpec],
    tool_names: dict[str, tuple[str, ...]],
    latency_profile: LatencyProfile,
    rng: random.Random,
    total_calls: int,
    payload_size: int,
) -> list[CallSpec]:
    provider_tool_pairs: list[tuple[ProviderSpec, str]] = []
    for spec in providers:
        for tool_name in tool_names[spec.provider.name]:
            provider_tool_pairs.append((spec, tool_name))

    calls: list[CallSpec] = []
    call_id = 0
    while len(calls) < total_calls:
        rng.shuffle(provider_tool_pairs)
        for spec, tool_name in provider_tool_pairs:
            if len(calls) >= total_calls:
                break
            delay = latency_profile.sample_delay(rng)
            arguments = {
                "query": f"call-{call_id}",
                "limit": call_id % 10,
                "payload": "x" * payload_size,
                "_benchmark_delay_sec": delay,
                "_benchmark_call_id": call_id,
            }
            calls.append(CallSpec(provider=spec.provider, tool_name=tool_name, arguments=arguments, delay_sec=delay))
            call_id += 1

    rng.shuffle(calls)
    return calls


def _chunk_calls(calls: list[CallSpec], batch_size: int) -> list[list[CallSpec]]:
    return [calls[idx : idx + batch_size] for idx in range(0, len(calls), batch_size)]


@contextlib.contextmanager
def _patch_mcp_io(registry: FakeServerRegistry) -> Iterator[None]:
    import data_designer.engine.mcp.io as mcp_io

    original_client_session = mcp_io.ClientSession
    original_sse_client = mcp_io.sse_client

    def fake_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> FakeServerContext:
        _ = headers
        return registry.context_for_endpoint(endpoint)

    mcp_io.ClientSession = FakeClientSession
    mcp_io.sse_client = fake_sse_client
    try:
        yield
    finally:
        mcp_io.ClientSession = original_client_session
        mcp_io.sse_client = original_sse_client


def _run_scenario(
    *,
    scenario_name: str,
    latency_profile: LatencyProfile,
    provider_count: int,
    tools_per_provider: int,
    total_calls: int,
    batch_size: int,
    payload_size: int,
    warmup_batches: int,
    seed: int,
) -> BenchmarkResult:
    rng = random.Random(seed)
    providers, tool_names = _build_provider_specs(provider_count, tools_per_provider)
    registry = FakeServerRegistry(providers, latency_profile, rng)

    warmup_calls_total = warmup_batches * batch_size
    all_calls = _build_call_specs(
        providers=providers,
        tool_names=tool_names,
        latency_profile=latency_profile,
        rng=rng,
        total_calls=total_calls + warmup_calls_total,
        payload_size=payload_size,
    )

    warmup_calls = all_calls[:warmup_calls_total]
    measured_calls = all_calls[warmup_calls_total:]
    warmup_batches_list = _chunk_calls(warmup_calls, batch_size) if warmup_calls else []
    measured_batches = _chunk_calls(measured_calls, batch_size)

    server_walltime_sec = sum(max(call.delay_sec for call in batch) for batch in measured_batches if batch)

    with _patch_mcp_io(registry):
        service = MCPIOService()
        try:
            for spec in providers:
                service.list_tools(spec.provider)
                service.list_tools(spec.provider)

            for batch in warmup_batches_list:
                service.call_tools([(call.provider, call.tool_name, call.arguments) for call in batch])

            tracemalloc.start()
            start_current = tracemalloc.get_traced_memory()[0]
            rss_start = _get_rss_bytes()
            start_time = time.perf_counter()

            for batch in measured_batches:
                service.call_tools([(call.provider, call.tool_name, call.arguments) for call in batch])

            elapsed_sec = time.perf_counter() - start_time
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rss_end = _get_rss_bytes()
        finally:
            service.shutdown()

    rpm = (total_calls / elapsed_sec * 60.0) if elapsed_sec > 0 else 0.0
    overhead_sec = max(0.0, elapsed_sec - server_walltime_sec)
    overhead_pct = (overhead_sec / elapsed_sec * 100.0) if elapsed_sec > 0 else 0.0

    rss_start_mb = (rss_start / BYTES_PER_MB) if rss_start is not None else None
    rss_end_mb = (rss_end / BYTES_PER_MB) if rss_end is not None else None
    rss_growth_mb = ((rss_end - rss_start) / BYTES_PER_MB) if rss_start is not None and rss_end is not None else None

    py_current_mb = current_bytes / BYTES_PER_MB
    py_peak_mb = peak_bytes / BYTES_PER_MB
    py_growth_mb = (current_bytes - start_current) / BYTES_PER_MB

    return BenchmarkResult(
        scenario_name=scenario_name,
        total_calls=total_calls,
        elapsed_sec=elapsed_sec,
        rpm=rpm,
        server_walltime_sec=server_walltime_sec,
        overhead_sec=overhead_sec,
        overhead_pct=overhead_pct,
        rss_start_mb=rss_start_mb,
        rss_end_mb=rss_end_mb,
        rss_growth_mb=rss_growth_mb,
        py_current_mb=py_current_mb,
        py_peak_mb=py_peak_mb,
        py_growth_mb=py_growth_mb,
    )


def _print_result(result: BenchmarkResult, provider_count: int, tools_per_provider: int, batch_size: int) -> None:
    print(f"\nScenario: {result.scenario_name}")
    print(
        f"- providers: {provider_count} | tools/provider: {tools_per_provider} | batch size: {batch_size} | "
        f"calls: {result.total_calls}"
    )
    print(
        f"- walltime: {result.elapsed_sec:.3f}s | server walltime: {result.server_walltime_sec:.3f}s | "
        f"overhead: {result.overhead_sec:.3f}s ({result.overhead_pct:.1f}%)"
    )
    print(f"- throughput: {result.rpm:,.1f} RPM")
    print(
        "- memory: "
        f"RSS start {_format_mb(result.rss_start_mb)}, "
        f"RSS end {_format_mb(result.rss_end_mb)}, "
        f"RSS growth {_format_mb(result.rss_growth_mb)}"
    )
    print(
        "- python alloc: "
        f"current {result.py_current_mb:.2f} MB, "
        f"peak {result.py_peak_mb:.2f} MB, "
        f"growth {result.py_growth_mb:.2f} MB"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MCPIOService overhead with mocked MCP servers.")
    parser.add_argument("--providers", type=int, default=100, help="Number of MCP providers to simulate.")
    parser.add_argument("--tools-per-provider", type=int, default=6, help="Number of tools per provider.")
    parser.add_argument("--total-calls", type=int, default=2000, help="Total tool calls per scenario.")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of tool calls per batch.")
    parser.add_argument("--payload-size", type=int, default=128, help="Size of payload argument for each call.")
    parser.add_argument("--warmup-batches", type=int, default=1, help="Warmup batches excluded from timing.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=("all", "instant", "long-tail"),
        help="Which latency profile to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profiles = {
        "instant": LatencyProfile(
            name="instant",
            fast_range_sec=(0.0, 0.001),
            tail_probability=0.0,
            tail_range_sec=(0.0, 0.0),
            list_tools_delay_sec=0.0005,
        ),
        "long-tail": LatencyProfile(
            name="long-tail",
            fast_range_sec=(0.001, 0.004),
            tail_probability=0.05,
            tail_range_sec=(0.05, 0.2),
            list_tools_delay_sec=0.001,
        ),
    }

    selected_profiles: list[LatencyProfile]
    if args.profile == "all":
        selected_profiles = [profiles["instant"], profiles["long-tail"]]
    else:
        selected_profiles = [profiles[args.profile]]

    for offset, profile in enumerate(selected_profiles):
        result = _run_scenario(
            scenario_name=profile.name,
            latency_profile=profile,
            provider_count=args.providers,
            tools_per_provider=args.tools_per_provider,
            total_calls=args.total_calls,
            batch_size=args.batch_size,
            payload_size=args.payload_size,
            warmup_batches=args.warmup_batches,
            seed=args.seed + offset,
        )
        _print_result(result, args.providers, args.tools_per_provider, args.batch_size)


if __name__ == "__main__":
    main()
