# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the Switchyard routes used by the Data Designer experiments."""

from __future__ import annotations

import argparse
import os
from typing import Any, cast

import uvicorn
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.backends.multi_llm_backend import build_multi_llm_backend
from switchyard.lib.cost_estimator import estimate_cost
from switchyard.lib.profiles import ProfileSwitchyard
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.deterministic_routing_config import DeterministicRoutingConfig
from switchyard.lib.profiles.random_routing import RandomRoutingConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.route_table import RouteTable
from switchyard.lib.route_table_builders import (
    build_deterministic_routing_switchyard,
    build_random_routing_switchyard,
    build_tier_passthrough_switchyard,
)
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import ChatRequest

DEFAULT_WEAK_MODEL = "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
DEFAULT_STRONG_MODEL = "aws/anthropic/bedrock-claude-sonnet-4-6"
DEFAULT_CLASSIFIER_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash"
DEFAULT_STRONG_PROBABILITY = 0.35


class DifficultyHintRouter:
    """Route a request using a prompt marker that stands in for request metadata."""

    def __init__(self, *, weak: LlmTarget, strong: LlmTarget) -> None:
        self._weak = weak
        self._strong = strong

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        target = self._weak if "[ROUTE_HINT=easy]" in str(request.body) else self._strong
        ctx.selected_target = target.id
        ctx.selected_model = target.model
        return request


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def build_target(target_id: str, model: str, *, base_url: str, api_key: str) -> LlmTarget:
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if "Nemotron-3-Nano" in model else None
    return LlmTarget(
        id=target_id,
        model=model,
        format=BackendFormat.OPENAI,
        base_url=base_url,
        api_key=api_key,
        timeout_secs=60,
        extra_body=extra_body,
    )


def build_routes() -> tuple[RouteTable, StatsAccumulator, dict[str, Any]]:
    base_url = require_env("NVIDIA_INFERENCE_BASE_URL")
    api_key = require_env("NVIDIA_INFERENCE_API_KEY")
    weak_model = os.environ.get("SWITCHYARD_WEAK_MODEL", DEFAULT_WEAK_MODEL)
    strong_model = os.environ.get("SWITCHYARD_STRONG_MODEL", DEFAULT_STRONG_MODEL)
    classifier_model = os.environ.get("SWITCHYARD_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)

    weak = build_target("weak", weak_model, base_url=base_url, api_key=api_key)
    strong = build_target("strong", strong_model, base_url=base_url, api_key=api_key)
    classifier = build_target("classifier", classifier_model, base_url=base_url, api_key=api_key)
    stats = StatsAccumulator()
    table = RouteTable()

    table.register("dd-weak", build_tier_passthrough_switchyard(weak, stats))
    table.register("dd-strong", build_tier_passthrough_switchyard(strong, stats))
    table.register(
        "dd-hinted",
        ProfileSwitchyard(
            ComponentChainProfile(
                request_processors=[DifficultyHintRouter(weak=weak, strong=strong)],
                backend=build_multi_llm_backend((strong, weak)),
                fallback_target_on_evict="strong",
            ).with_runtime_components(stats_accumulator=stats)
        ),
    )
    table.register(
        "dd-smart",
        build_deterministic_routing_switchyard(
            DeterministicRoutingConfig(
                strong=strong,
                weak=weak,
                classifier=classifier,
                fallback_target_on_evict="strong",
                profile_name="general",
                classifier_min_confidence=0.5,
                classifier_timeout_s=60,
            ),
            stats,
        ),
    )
    table.register(
        "dd-mixed",
        build_random_routing_switchyard(
            RandomRoutingConfig(
                strong=strong,
                weak=weak,
                fallback_target_on_evict="strong",
                strong_probability=DEFAULT_STRONG_PROBABILITY,
                rng_seed=42,
            ),
            stats,
        ),
    )
    table.set_default_model("dd-smart")
    config = {
        "weak_model": weak_model,
        "strong_model": strong_model,
        "classifier_model": classifier_model,
        "mixed_strong_probability": DEFAULT_STRONG_PROBABILITY,
        "hinted_policy": "easy=weak, otherwise=strong",
    }
    return table, stats, config


def estimate_snapshot_cost(snapshot: dict[str, Any]) -> dict[str, Any]:
    backend_models = cast(dict[str, dict[str, object]], snapshot.get("models", {}))
    classifier = cast(dict[str, Any], snapshot.get("classifier", {}))
    classifier_models = cast(dict[str, dict[str, object]], classifier.get("models", {}))
    backend = estimate_cost(backend_models)
    classifier_cost = estimate_cost(classifier_models)
    return {
        "backend": backend,
        "classifier": classifier_cost,
        "total_cost": round(float(backend["total_cost"]) + float(classifier_cost["total_cost"]), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4011)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, stats, config = build_routes()
    app = build_switchyard_app(table)

    async def cost_estimate() -> dict[str, Any]:
        return estimate_snapshot_cost(stats.snapshot_sync())

    async def demo_config() -> dict[str, Any]:
        return config

    app.add_api_route("/demo/cost-estimate", cost_estimate, methods=["GET"])
    app.add_api_route("/demo/config", demo_config, methods=["GET"])
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
