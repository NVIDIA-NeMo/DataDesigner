# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from data_designer.config.base import ConfigBase, SingleColumnConfig
from data_designer.config.seed_source import HuggingFaceSeedSource
from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionResponse, ToolCall
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.plugins.plugin import Plugin, PluginType

MODULE_NAME = __name__


class StubHuggingFaceSeedReader(SeedReader[HuggingFaceSeedSource]):
    """Stub seed reader for testing."""

    def get_column_names(self) -> list[str]:
        return ["age", "city"]

    def get_dataset_uri(self) -> str:
        return "unused in these tests"

    def create_duckdb_connection(self) -> None:
        pass


class ValidTestConfig(SingleColumnConfig):
    """Valid config for testing plugin creation."""

    column_type: Literal["test-generator"] = "test-generator"
    name: str


class ValidTestTask(ColumnGeneratorCellByCell[ValidTestConfig]):
    """Valid task for testing plugin creation."""

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class ConfigWithoutDiscriminator(ConfigBase):
    some_field: str


class ConfigWithStringField(ConfigBase):
    column_type: str = "test-generator"


class ConfigWithNonStringDefault(ConfigBase):
    column_type: Literal["test-generator"] = 123  # type: ignore


class ConfigWithInvalidKey(ConfigBase):
    column_type: Literal["invalid-key-!@#"] = "invalid-key-!@#"


class StubPluginConfigA(SingleColumnConfig):
    column_type: Literal["test-plugin-a"] = "test-plugin-a"


class StubPluginConfigB(SingleColumnConfig):
    column_type: Literal["test-plugin-b"] = "test-plugin-b"


class StubPluginTaskA(ColumnGeneratorCellByCell[StubPluginConfigA]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class StubPluginTaskB(ColumnGeneratorCellByCell[StubPluginConfigB]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


# Stub plugins requiring different combinations of resources


class StubPluginConfigModels(SingleColumnConfig):
    column_type: Literal["test-plugin-models"] = "test-plugin-models"


class StubPluginConfigModelsAndBlobs(SingleColumnConfig):
    column_type: Literal["test-plugin-models-and-blobs"] = "test-plugin-models-and-blobs"


class StubPluginConfigBlobsAndSeeds(SingleColumnConfig):
    column_type: Literal["test-plugin-blobs-and-seeds"] = "test-plugin-blobs-and-seeds"


class StubPluginTaskModels(ColumnGeneratorCellByCell[StubPluginConfigModels]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class StubPluginTaskModelsAndBlobs(ColumnGeneratorCellByCell[StubPluginConfigModelsAndBlobs]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class StubPluginTaskBlobsAndSeeds(ColumnGeneratorCellByCell[StubPluginConfigBlobsAndSeeds]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


plugin_none = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigA",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskA",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugin_models = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigModels",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskModels",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugin_models_and_blobs = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigModelsAndBlobs",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskModelsAndBlobs",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugin_blobs_and_seeds = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigBlobsAndSeeds",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskBlobsAndSeeds",
    plugin_type=PluginType.COLUMN_GENERATOR,
)


# =============================================================================
# Stub LLM response classes for testing (legacy, kept for backward compat)
# =============================================================================


class StubMessage:
    """Stub message class for mocking LLM completion responses.

    Args:
        content: The message content.
        tool_calls: Optional list of tool call dictionaries.
        reasoning_content: Optional reasoning content for reasoning models.
    """

    def __init__(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class StubChoice:
    """Stub choice class for mocking LLM completion responses."""

    def __init__(self, message: StubMessage) -> None:
        self.message = message


class StubResponse:
    """Stub response class for mocking LLM completion responses."""

    def __init__(self, message: StubMessage) -> None:
        self.choices = [StubChoice(message)]


# =============================================================================
# Canonical stub helpers
# =============================================================================


def make_stub_completion_response(
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> ChatCompletionResponse:
    """Factory helper for creating canonical ChatCompletionResponse test objects."""
    return ChatCompletionResponse(
        message=AssistantMessage(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or [],
        ),
    )


# =============================================================================
# Stub MCP classes for testing tool calling
# =============================================================================


class StubMCPFacade:
    """Configurable stub MCP facade for testing tool calling functionality.

    This stub provides a configurable implementation of the MCP facade interface
    for testing model facade tool calling behavior without real MCP connections.

    Args:
        max_tool_call_turns: Maximum number of tool call turns allowed before refusal.
        tool_schemas: List of tool schemas to return from get_tool_schemas().
        process_fn: Custom function to process completion responses with tool calls.
        refuse_fn: Custom function to generate refusal messages when budget exceeded.
    """

    def __init__(
        self,
        max_tool_call_turns: int = 3,
        tool_schemas: list[dict[str, Any]] | None = None,
        process_fn: Callable[[ChatCompletionResponse], list[ChatMessage]] | None = None,
        refuse_fn: Callable[[ChatCompletionResponse], list[ChatMessage]] | None = None,
    ) -> None:
        self.tool_alias = "tools"
        self.providers = ["tools"]
        self.max_tool_call_turns = max_tool_call_turns
        self._tool_schemas = tool_schemas or [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        self._process_fn = process_fn
        self._refuse_fn = refuse_fn

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return self._tool_schemas

    def get_tool_call_count(self, completion_response: ChatCompletionResponse) -> int:
        return len(completion_response.message.tool_calls)

    def has_tool_calls(self, completion_response: ChatCompletionResponse) -> bool:
        return len(completion_response.message.tool_calls) > 0

    def process_completion_response(self, completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        if self._process_fn:
            return self._process_fn(completion_response)
        message = completion_response.message
        tool_calls = message.tool_calls
        tool_call_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments_json},
            }
            for tc in tool_calls
        ]
        return [
            ChatMessage.as_assistant(content=message.content or "", tool_calls=tool_call_dicts),
            *[ChatMessage.as_tool(content="tool-result", tool_call_id=tc.id) for tc in tool_calls],
        ]

    def refuse_completion_response(self, completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        if self._refuse_fn:
            return self._refuse_fn(completion_response)
        message = completion_response.message
        tool_calls = message.tool_calls
        tool_call_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments_json},
            }
            for tc in tool_calls
        ]
        return [
            ChatMessage.as_assistant(content="", tool_calls=tool_call_dicts),
            *[
                ChatMessage.as_tool(
                    content="Tool call refused: maximum tool-calling turns reached.",
                    tool_call_id=tc.id,
                )
                for tc in tool_calls
            ],
        ]


class StubMCPRegistry:
    """Stub MCP registry that returns a configurable StubMCPFacade."""

    def __init__(self, facade: StubMCPFacade | None = None) -> None:
        self._facade = facade or StubMCPFacade()

    def get_mcp(self, *, tool_alias: str) -> StubMCPFacade:
        return self._facade
