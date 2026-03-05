# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from data_designer.config.mcp import MCPProviderT, ToolConfig
from data_designer.engine.mcp import io as mcp_io
from data_designer.engine.mcp.errors import DuplicateToolNameError, MCPConfigurationError, MCPToolError
from data_designer.engine.mcp.registry import MCPToolDefinition
from data_designer.engine.model_provider import MCPProviderRegistry
from data_designer.engine.models.clients.types import ChatCompletionResponse, ToolCall
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.secret_resolver import SecretResolver

DEFAULT_TOOL_REFUSAL_MESSAGE = (
    "Tool call refused: You have reached the maximum number of tool-calling turns. "
    "Please provide your final response without requesting additional tool calls."
)


class MCPFacade:
    """Lightweight facade scoped to a specific ToolConfig.

    MCPFacade provides a clean interface for MCP tool operations within the context
    of a specific tool configuration. It handles tool call extraction, validation,
    and execution using the mcp.io module for communication.

    This mirrors the ModelFacade pattern where each facade is scoped to a specific
    configuration while sharing underlying resources through caching in the io module.
    """

    def __init__(
        self,
        tool_config: ToolConfig,
        secret_resolver: SecretResolver,
        mcp_provider_registry: MCPProviderRegistry,
    ) -> None:
        self._tool_config = tool_config
        self._secret_resolver = secret_resolver
        self._mcp_provider_registry = mcp_provider_registry

    @property
    def tool_alias(self) -> str:
        """The alias for this tool configuration."""
        return self._tool_config.tool_alias

    @property
    def providers(self) -> list[str]:
        """List of MCP provider names for this configuration."""
        return self._tool_config.providers

    @property
    def max_tool_call_turns(self) -> int:
        """Maximum number of tool-calling turns permitted in a single generation.

        A turn is one iteration where the LLM requests tool calls. With parallel
        tool calling, a single turn may execute multiple tools simultaneously.
        """
        return self._tool_config.max_tool_call_turns

    @property
    def allow_tools(self) -> list[str] | None:
        """Optional allowlist of permitted tool names."""
        return self._tool_config.allow_tools

    @property
    def timeout_sec(self) -> float | None:
        """Timeout in seconds for MCP tool calls."""
        return self._tool_config.timeout_sec

    @staticmethod
    def get_tool_call_count(completion_response: ChatCompletionResponse) -> int:
        """Count the number of tool calls in a completion response."""
        return len(completion_response.message.tool_calls)

    @staticmethod
    def has_tool_calls(completion_response: ChatCompletionResponse) -> bool:
        """Returns True if tool calls are present in the completion response."""
        return len(completion_response.message.tool_calls) > 0

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for this configuration.

        Fetches tools from all providers in the configuration and applies
        allow_tools filtering if specified. Uses cached results from mcp_io.

        Returns:
            List of tool schemas in OpenAI function calling format.

        Raises:
            MCPConfigurationError: If allowed tools are not found on any provider.
            DuplicateToolNameError: If the same tool name appears in multiple providers.
        """
        all_tools: list[MCPToolDefinition] = []
        tool_to_providers: dict[str, list[str]] = {}

        for provider_name in self._tool_config.providers:
            provider = self._mcp_provider_registry.get_provider(provider_name)
            resolved_provider = self._resolve_provider(provider)
            tools = mcp_io.list_tools(
                resolved_provider, timeout_sec=self._tool_config.timeout_sec
            )  # Cached in io module
            for tool in tools:
                tool_to_providers.setdefault(tool.name, []).append(provider_name)
            all_tools.extend(tools)

        # Check for duplicate tool names across providers
        duplicates = {name: providers for name, providers in tool_to_providers.items() if len(providers) > 1}
        if duplicates:
            dup_details = [f"'{name}' (in: {', '.join(providers)})" for name, providers in sorted(duplicates.items())]
            raise DuplicateToolNameError(
                f"Duplicate tool names found across MCP providers: {'; '.join(dup_details)}. "
                "Each tool name must be unique across all providers in a ToolConfig."
            )

        all_available_names = set(tool_to_providers.keys())
        allowed_names = set(self._tool_config.allow_tools) if self._tool_config.allow_tools else None
        if allowed_names is not None:
            missing = allowed_names.difference(all_available_names)
            if missing:
                provider_list = ", ".join(repr(p) for p in self._tool_config.providers)
                raise MCPConfigurationError(
                    f"Tool(s) {sorted(missing)!r} not found on any of the MCP providers: {provider_list}."
                )
            all_tools = [tool for tool in all_tools if tool.name in allowed_names]

        return [tool.to_openai_tool_schema() for tool in all_tools]

    def process_completion_response(
        self,
        completion_response: ChatCompletionResponse,
    ) -> list[ChatMessage]:
        """Process an LLM completion response and execute any tool calls.

        This is the primary method for handling tool calls from an LLM response.
        It extracts the response content, reasoning content, and all tool calls
        from the completion response, executes each tool call (including parallel
        tool calls), and returns the messages for continuing the conversation.

        Args:
            completion_response: The canonical ChatCompletionResponse from the model client.

        Returns:
            A list of ChatMessages to append to the conversation history:
            - If tool calls were present: [assistant_message_with_tool_calls, *tool_response_messages]
            - If no tool calls: [assistant_message]

        Raises:
            MCPToolError: If a requested tool is not in the allowed tools list.
            MCPToolError: If tool execution fails or times out.
            MCPConfigurationError: If a requested tool is not found on any configured provider.
        """
        message = completion_response.message

        response_content = message.content or ""
        reasoning_content = message.reasoning_content

        # Strip whitespace if reasoning is present (models often add extra newlines)
        if reasoning_content:
            response_content = response_content.strip()
            reasoning_content = reasoning_content.strip()

        tool_calls = message.tool_calls

        if not tool_calls:
            return [
                ChatMessage.as_assistant(
                    content=response_content,
                    reasoning_content=reasoning_content or None,
                )
            ]

        # Has tool calls - execute and return all messages
        tool_call_dicts = _convert_canonical_tool_calls_to_dicts(tool_calls)
        assistant_message = self._build_assistant_tool_message(response_content, tool_call_dicts, reasoning_content)
        tool_messages = self._execute_tool_calls_from_canonical(tool_calls)

        return [assistant_message, *tool_messages]

    def refuse_completion_response(
        self,
        completion_response: ChatCompletionResponse,
        refusal_message: str | None = None,
    ) -> list[ChatMessage]:
        """Refuse tool calls without executing them.

        Used when the tool call turn budget is exhausted. Returns messages
        that include the assistant's tool call request but with refusal
        responses instead of actual tool results.

        Args:
            completion_response: The canonical ChatCompletionResponse containing tool calls.
            refusal_message: Optional custom refusal message.

        Returns:
            A list of ChatMessages to append to the conversation history.
        """
        message = completion_response.message

        response_content = message.content or ""
        reasoning_content = message.reasoning_content

        # Strip whitespace if reasoning is present (models often add extra newlines)
        if reasoning_content:
            response_content = response_content.strip()
            reasoning_content = reasoning_content.strip()

        tool_calls = message.tool_calls

        if not tool_calls:
            return [
                ChatMessage.as_assistant(
                    content=response_content,
                    reasoning_content=reasoning_content or None,
                )
            ]

        # Build assistant message with tool calls (same as normal)
        tool_call_dicts = _convert_canonical_tool_calls_to_dicts(tool_calls)
        assistant_message = self._build_assistant_tool_message(response_content, tool_call_dicts, reasoning_content)

        # Build refusal messages instead of executing tools
        refusal = refusal_message or DEFAULT_TOOL_REFUSAL_MESSAGE
        tool_messages = [ChatMessage.as_tool(content=refusal, tool_call_id=tc.id) for tc in tool_calls]

        return [assistant_message, *tool_messages]

    def _resolve_provider(self, provider: MCPProviderT) -> MCPProviderT:
        """Resolve secret references in an MCP provider's api_key."""
        api_key_ref = getattr(provider, "api_key", None)
        if not api_key_ref:
            return provider
        resolved_key = self._secret_resolver.resolve(api_key_ref)
        return provider.model_copy(update={"api_key": resolved_key})

    def _build_assistant_tool_message(
        self,
        response: str | None,
        tool_calls: list[dict[str, Any]],
        reasoning_content: str | None = None,
    ) -> ChatMessage:
        """Build the assistant message containing tool call requests."""
        tool_calls_payload = [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {"name": tool_call["name"], "arguments": tool_call["arguments_json"]},
            }
            for tool_call in tool_calls
        ]
        return ChatMessage.as_assistant(
            content=response or "",
            reasoning_content=reasoning_content or None,
            tool_calls=tool_calls_payload,
        )

    def _execute_tool_calls_from_canonical(
        self,
        tool_calls: list[ToolCall],
    ) -> list[ChatMessage]:
        """Execute canonical ToolCall objects and return tool response messages."""
        allowed_tools = set(self._tool_config.allow_tools) if self._tool_config.allow_tools else None

        calls_to_execute: list[tuple[MCPProviderT, str, dict[str, Any], str]] = []
        for tc in tool_calls:
            if allowed_tools is not None and tc.name not in allowed_tools:
                providers_str = ", ".join(repr(p) for p in self._tool_config.providers)
                raise MCPToolError(f"Tool {tc.name!r} is not permitted for providers: {providers_str}.")

            arguments = json.loads(tc.arguments_json) if tc.arguments_json else {}
            resolved_provider = self._find_resolved_provider_for_tool(tc.name)
            calls_to_execute.append((resolved_provider, tc.name, arguments, tc.id))

        # Execute all calls in parallel
        results = mcp_io.call_tools(
            [(p, n, a) for p, n, a, _ in calls_to_execute],
            timeout_sec=self._tool_config.timeout_sec,
        )

        return [
            ChatMessage.as_tool(content=result.content, tool_call_id=call[3])
            for result, call in zip(results, calls_to_execute)
        ]

    def _find_resolved_provider_for_tool(self, tool_name: str) -> MCPProviderT:
        """Find the provider that has the given tool and return it with resolved api_key."""
        for provider_name in self._tool_config.providers:
            provider = self._mcp_provider_registry.get_provider(provider_name)
            resolved_provider = self._resolve_provider(provider)
            tools = mcp_io.list_tools(
                resolved_provider, timeout_sec=self._tool_config.timeout_sec
            )  # Cached in io module
            if any(tool.name == tool_name for tool in tools):
                return resolved_provider

        raise MCPConfigurationError(f"Tool {tool_name!r} not found on any configured provider.")


def _convert_canonical_tool_calls_to_dicts(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Convert canonical ToolCall objects to the internal dict format for ChatMessage."""
    return [
        {
            "id": tc.id,
            "name": tc.name,
            "arguments_json": tc.arguments_json,
        }
        for tc in tool_calls
    ]
