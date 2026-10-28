# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import AsyncExitStack

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import ChatCompletionRequest
from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController
from data_designer.engine.models.request_admission.resources import RequestDomain, RequestResourceKey
from data_designer.slurm.runtime.proxy import _Backend, _ProxyApplication


@pytest.mark.asyncio
@pytest.mark.parametrize("total_overload", (False, True), ids=("partial-overload", "total-overload"))
async def test_proxy_overload_reaches_real_client_admission(total_overload: bool) -> None:
    calls: list[int] = []

    def application(index: int) -> web.Application:
        async def respond(request: web.Request) -> web.Response:
            calls.append(index)
            if total_overload or index == 0:
                return web.json_response({"error": {"message": "overloaded"}}, status=429, headers={"Retry-After": "3"})
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "complete"}}]})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", respond)
        return app

    async with AsyncExitStack() as stack:
        servers = [await stack.enter_async_context(TestServer(application(index))) for index in range(8)]
        proxy = _ProxyApplication(tuple(_Backend(server.host, server.port) for server in servers), 1)
        endpoint = await stack.enter_async_context(TestServer(proxy.create()))
        resource = RequestResourceKey("slurm", "model", RequestDomain.CHAT)
        admission = AdaptiveRequestAdmissionController(
            RequestAdmissionConfig(multiplicative_decrease_factor=0.5, cooldown_seconds=0)
        )
        admission.register(provider_name="slurm", model_id="model", alias="generator", max_parallel_requests=8)
        client = OpenAICompatibleClient(
            provider_name="slurm",
            endpoint=str(endpoint.make_url("/v1")),
            concurrency_mode=ClientConcurrencyMode.ASYNC,
            retry_config=RetryConfig(max_retries=0),
        )
        stack.push_async_callback(client.aclose)
        executor = ModelRequestExecutor(client, admission, "slurm", "model", retry_config=RetryConfig(max_retries=0))
        request = ChatCompletionRequest(model="model", messages=[])
        if total_overload:
            with pytest.raises(ProviderError) as failure:
                await executor.acompletion(request)
            assert failure.value.kind is ProviderErrorKind.RATE_LIMIT
            assert failure.value.retry_after == 3
            assert calls == [0, 1, 2]
        else:
            assert (await executor.acompletion(request)).message.content == "complete"
            assert calls == [0, 1]
        pressure = admission.pressure.snapshot(resource)
        assert pressure is not None
        assert pressure.current_limit == (4 if total_overload else 8)
        assert pressure.last_outcome == ("rate_limited" if total_overload else "success")
        if total_overload:
            assert pressure.cooldown_remaining_seconds > 2
        else:
            assert pressure.cooldown_remaining_seconds == 0
        assert pressure.in_flight_count == 0
