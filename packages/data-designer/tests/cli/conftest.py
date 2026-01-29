# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRegistry, MCPProviderRepository
from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.cli.repositories.tool_repository import ToolConfigRegistry, ToolRepository
from data_designer.cli.services.mcp_provider_service import MCPProviderService
from data_designer.cli.services.model_service import ModelService
from data_designer.cli.services.provider_service import ProviderService
from data_designer.cli.services.tool_service import ToolService
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, ToolConfig

MCPProviderT = MCPProvider | LocalStdioMCPProvider
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider


@pytest.fixture
def stub_inference_parameters() -> ChatCompletionInferenceParams:
    return ChatCompletionInferenceParams(temperature=0.7, top_p=0.9, max_tokens=2048, max_parallel_requests=4)


@pytest.fixture
def stub_model_configs(stub_inference_parameters: ChatCompletionInferenceParams) -> list[ModelConfig]:
    return [
        ModelConfig(
            alias="test-alias-1",
            model="test-model-1",
            provider="test-provider-1",
            inference_parameters=stub_inference_parameters,
        ),
        ModelConfig(
            alias="test-alias-2",
            model="test-model-2",
            provider="test-provider-1",
            inference_parameters=stub_inference_parameters,
        ),
    ]


@pytest.fixture
def stub_new_model_config() -> ModelConfig:
    return ModelConfig(
        alias="test-alias-3",
        model="test-model-3",
        provider="test-provider-1",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            max_parallel_requests=4,
            timeout=100,
        ),
    )


@pytest.fixture
def stub_model_providers() -> list[ModelProvider]:
    return [
        ModelProvider(
            name="test-provider-1",
            endpoint="https://api.example.com/v1",
            provider_type="openai",
            api_key="test-api-key",
        ),
        ModelProvider(
            name="test-provider-2",
            endpoint="https://api.example.com/v2",
            provider_type="openai",
            api_key="test-api-key-2",
        ),
    ]


@pytest.fixture
def stub_new_model_provider() -> ModelProvider:
    return ModelProvider(
        name="test-provider-3",
        endpoint="https://api.example.com/v1",
        provider_type="openai",
        api_key="test-api-key-1",
    )


@pytest.fixture
def stub_model_service(tmp_path: Path, stub_model_configs: list[ModelConfig]) -> ModelService:
    repository = ModelRepository(tmp_path)
    repository.save(ModelConfigRegistry(model_configs=stub_model_configs))
    return ModelService(repository)


@pytest.fixture
def stub_provider_service(tmp_path: Path, stub_model_providers: list[ModelProvider]) -> ProviderService:
    repository = ProviderRepository(tmp_path)
    repository.save(ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name))
    return ProviderService(repository)


@pytest.fixture
def stub_mcp_providers() -> list[MCPProviderT]:
    return [
        MCPProvider(
            name="mcp-provider-1",
            endpoint="http://localhost:8080/sse",
            api_key="mcp-api-key-1",
        ),
        MCPProvider(
            name="mcp-provider-2",
            endpoint="http://localhost:8081/sse",
            api_key=None,
        ),
        LocalStdioMCPProvider(
            name="mcp-provider-stdio",
            command="python",
            args=["-m", "my_mcp_server"],
            env={"PYTHONPATH": "/app"},
        ),
    ]


@pytest.fixture
def stub_new_mcp_provider() -> MCPProvider:
    return MCPProvider(
        name="mcp-provider-3",
        endpoint="http://localhost:8082/sse",
        api_key="mcp-api-key-3",
    )


@pytest.fixture
def stub_new_stdio_mcp_provider() -> LocalStdioMCPProvider:
    return LocalStdioMCPProvider(
        name="mcp-provider-stdio-new",
        command="node",
        args=["server.js", "--port", "3000"],
        env={"NODE_ENV": "production"},
    )


@pytest.fixture
def stub_mcp_provider_service(tmp_path: Path, stub_mcp_providers: list[MCPProviderT]) -> MCPProviderService:
    repository = MCPProviderRepository(tmp_path)
    repository.save(MCPProviderRegistry(providers=stub_mcp_providers))
    return MCPProviderService(repository)


@pytest.fixture
def stub_tool_configs() -> list[ToolConfig]:
    return [
        ToolConfig(
            tool_alias="tool-config-1",
            providers=["mcp-provider-1"],
            allow_tools=["tool-a", "tool-b"],
            max_tool_call_turns=5,
            timeout_sec=30.0,
        ),
        ToolConfig(
            tool_alias="tool-config-2",
            providers=["mcp-provider-1", "mcp-provider-2"],
            allow_tools=None,
            max_tool_call_turns=10,
        ),
    ]


@pytest.fixture
def stub_new_tool_config() -> ToolConfig:
    return ToolConfig(
        tool_alias="tool-config-3",
        providers=["mcp-provider-2"],
        allow_tools=["tool-c"],
        max_tool_call_turns=3,
    )


@pytest.fixture
def stub_tool_service(tmp_path: Path, stub_tool_configs: list[ToolConfig]) -> ToolService:
    repository = ToolRepository(tmp_path)
    repository.save(ToolConfigRegistry(tool_configs=stub_tool_configs))
    return ToolService(repository)
