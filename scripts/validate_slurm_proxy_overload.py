# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate real vLLM queue overload through the production proxy and DD client.

Run inside an allocation with two idle vLLM replicas, each configured with
max-num-seqs=1, DD_VLLM_MAX_WAITING_REQUESTS=0, and DD_VLLM_RETRY_AFTER_SECONDS=3.
Both replicas must load the package's QueueDepthBackpressureMiddleware. No model,
image, scheduler, or cluster-specific paths are embedded in this harness.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from data_designer.config import DataDesignerConfigBuilder
from data_designer.config.models import ModelProvider
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import ChatCompletionRequest
from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController
from data_designer.engine.models.request_admission.resources import RequestDomain, RequestResourceKey
from data_designer.interface import DataDesigner


async def queue_depth(client: ClientSession, backend: str) -> float:
    """Read actual vLLM waiting-request metrics without synthesizing overload."""
    async with client.get(f"{backend}/metrics") as response:
        response.raise_for_status()
        text = await response.text()
    samples = re.findall(r"^vllm:num_requests_waiting(?:\{[^\n]*\})?\s+([\d.eE+-]+)", text, re.MULTILINE)
    if not samples:
        raise RuntimeError("required vLLM queue metric is unavailable")
    return sum(float(value) for value in samples)


@asynccontextmanager
async def saturate(client: ClientSession, backends: list[str], model: str) -> AsyncIterator[list[float]]:
    """Queue real long completions and cancel only this harness's requests."""

    async def generate(backend: str) -> int:
        async with client.post(
            f"{backend}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Continue counting integers, one per line."}],
                "max_tokens": 1024,
                "min_tokens": 1024,
                "ignore_eos": True,
                "temperature": 0,
            },
        ) as response:
            await response.read()
            return response.status

    pending = [asyncio.create_task(generate(backend)) for backend in backends for _ in range(8)]
    try:
        deadline = time.monotonic() + 20
        while True:
            depths = [await queue_depth(client, backend) for backend in backends]
            if all(depth > 0 for depth in depths):
                await asyncio.sleep(0.2)  # Allow the production 100 ms sampler to observe the queue.
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("real backend queues did not saturate within the bounded window")
            await asyncio.sleep(0.05)
        yield depths
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        deadline = time.monotonic() + 30
        while any([await queue_depth(client, backend) for backend in backends]):
            if time.monotonic() >= deadline:
                raise RuntimeError("backend queues did not recover after cancelling validation traffic")
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)


def proxy_usage(pid: int) -> dict[str, float]:
    """Read Linux process CPU and peak resident memory for the proxy alone."""
    status = Path(f"/proc/{pid}/status").read_text()
    peak = re.search(r"^VmHWM:\s+(\d+) kB", status, re.MULTILINE)
    fields = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
    return {
        "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
        "peak_rss_mib": int(peak.group(1)) / 1024 if peak else 0,
    }


@asynccontextmanager
async def proxy_process(backends: list[str], directory: Path) -> AsyncIterator[tuple[str, subprocess.Popen[bytes]]]:
    """Launch the installed production entrypoint with private raw logs."""
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [sys.executable, "-m", "data_designer.slurm.runtime.proxy", "--listen-port", str(port)]
    command.extend(argument for backend in backends for argument in ("--backend", backend))
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "proxy.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            origin = f"http://127.0.0.1:{port}"
            async with ClientSession(timeout=ClientTimeout(total=8)) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError("production proxy exited before readiness")
                    try:
                        async with client.get(f"{origin}/health") as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("production proxy did not become ready")
            yield origin, process
        finally:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)


def generate_dataset(origin: str, model: str, directory: Path, records: int) -> dict[str, object]:
    """Generate and verify actual persisted Data Designer output."""
    builder = DataDesignerConfigBuilder.from_config(
        {
            "model_configs": [
                {
                    "alias": "generator",
                    "model": model,
                    "provider": "live",
                    "inference_parameters": {
                        "max_tokens": 256,
                        "temperature": 0,
                        "max_parallel_requests": 1,
                        "extra_body": {"reasoning_effort": "low"},
                    },
                }
            ],
            "columns": [
                {
                    "name": "answer",
                    "column_type": "llm-text",
                    "model_alias": "generator",
                    "prompt": "Reply with only the word READY. Do not explain.",
                }
            ],
        }
    )
    designer = DataDesigner(
        artifact_path=directory,
        model_providers=[ModelProvider(name="live", endpoint=f"{origin}/v1")],
        auto_configure_logging=False,
    )
    started = time.monotonic()
    results = designer.create(builder, num_records=records, dataset_name="live-overload-acceptance")
    dataset = results.load_dataset()
    assert len(dataset) == records
    assert dataset["answer"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    elapsed = time.monotonic() - started
    return {
        "expected_records": records,
        "actual_records": len(dataset),
        "nonempty_answers": True,
        "elapsed_seconds": elapsed,
        "records_per_minute": records * 60 / elapsed,
    }


async def validate(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute healthy, partial, and total overload plus cooldown recovery."""
    backends: list[str] = arguments.backend
    assert len(backends) == 2, "this bounded acceptance scenario requires exactly two replicas"
    assert re.fullmatch(r"[0-9a-f]{40}", arguments.source_commit)
    report: dict[str, Any] = {"source_commit": arguments.source_commit, "replicas": 2, "cases": {}}
    report["wheel_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(arguments.wheels.glob("*.whl"))
    }
    assert len(report["wheel_sha256"]) == 4
    report["installed_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("data-designer", "data-designer-config", "data-designer-engine", "data-designer-slurm")
    }
    async with ClientSession(timeout=ClientTimeout(total=180)) as client:
        async with client.get(f"{backends[0]}/version") as response:
            report["vllm_version"] = (await response.json())["version"]
        async with proxy_process(backends, arguments.output.parent / "healthy") as (origin, process):
            report["cases"]["healthy_generation"] = await asyncio.to_thread(
                generate_dataset, origin, arguments.model, arguments.output.parent / "dataset", arguments.records
            )
            async with client.get(f"{origin}/metrics") as response:
                metrics = await response.json()
            assert metrics["connections"]["reused"] > 0
            report["cases"]["healthy_generation"].update(
                {"proxy_metrics": metrics, "proxy_usage": proxy_usage(process.pid)}
            )

        request = ChatCompletionRequest(
            model=arguments.model,
            messages=[{"role": "user", "content": "Say READY."}],
            max_tokens=256,
            extra_body={"reasoning_effort": "low"},
        )
        for total in (False, True):
            label = "total_overload" if total else "partial_overload"
            async with proxy_process(backends, arguments.output.parent / label) as (origin, process):
                admission = AdaptiveRequestAdmissionController(
                    RequestAdmissionConfig(multiplicative_decrease_factor=0.5, cooldown_seconds=0)
                )
                admission.register(
                    provider_name="live", model_id=arguments.model, alias="generator", max_parallel_requests=8
                )
                adapter = OpenAICompatibleClient(
                    provider_name="live",
                    endpoint=f"{origin}/v1",
                    concurrency_mode=ClientConcurrencyMode.ASYNC,
                    retry_config=RetryConfig(max_retries=0),
                )
                executor = ModelRequestExecutor(
                    adapter, admission, "live", arguments.model, retry_config=RetryConfig(max_retries=0)
                )
                try:
                    async with saturate(client, backends if total else backends[:1], arguments.model) as depths:
                        async with client.get(f"{origin}/health") as response:
                            assert response.status == 200, "overload must not cause health failure"
                        if total:
                            try:
                                await executor.acompletion(request)
                            except ProviderError as error:
                                assert error.kind is ProviderErrorKind.RATE_LIMIT
                                assert error.retry_after == 3
                            else:
                                raise AssertionError("real total overload did not reach client admission as 429")
                        else:
                            result = await executor.acompletion(request)
                            assert result.message.content or result.message.reasoning_content
                        pressure = admission.pressure.snapshot(
                            RequestResourceKey("live", arguments.model, RequestDomain.CHAT)
                        )
                        assert pressure is not None and pressure.current_limit == (4 if total else 8)
                        assert (
                            pressure.cooldown_remaining_seconds > 2
                            if total
                            else pressure.cooldown_remaining_seconds == 0
                        )
                        async with client.get(f"{origin}/metrics") as response:
                            metrics = await response.json()
                        assert metrics["attempts"] == 2
                        assert metrics["retries"] == {"http_429": 1}
                        report["cases"][label] = {
                            "observed_waiting_requests": depths,
                            "admission_limit": pressure.current_limit,
                            "cooldown_remaining_seconds": pressure.cooldown_remaining_seconds,
                            "proxy_metrics": metrics,
                        }
                    remaining = admission.pressure.snapshot(
                        RequestResourceKey("live", arguments.model, RequestDomain.CHAT)
                    ).cooldown_remaining_seconds
                    started = time.monotonic()
                    result = await executor.acompletion(request)
                    elapsed = time.monotonic() - started
                    assert result.message.content or result.message.reasoning_content
                    assert elapsed >= remaining - 0.05
                    report["cases"][label].update(
                        {
                            "recovery_succeeded": True,
                            "recovery_elapsed_seconds": elapsed,
                            "cooldown_remaining_before_recovery_seconds": remaining,
                            "proxy_usage": proxy_usage(process.pid),
                        }
                    )
                finally:
                    await adapter.aclose()
    report["passed"] = True
    return report


def main() -> None:
    """Run explicitly configured live acceptance and write a sanitized summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=32)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = asyncio.run(validate(arguments))
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": list(report["cases"])}))


if __name__ == "__main__":
    main()
