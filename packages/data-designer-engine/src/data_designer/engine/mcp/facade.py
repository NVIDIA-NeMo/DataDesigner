# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import uuid
from typing import Any

from data_designer.config.mcp import MCPProviderT, ToolConfig
from data_designer.engine.mcp import io as mcp_io
from data_designer.engine.mcp.errors import DuplicateToolNameError, MCPConfigurationError, MCPToolError
from data_designer.engine.mcp.registry import MCPToolDefinition
from data_designer.engine.model_provider import MCPProviderRegistry
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
        """Initialize the MCPFacade.

        Args:
            tool_config: The tool configuration this facade is scoped to.
            secret_resolver: Resolver for secrets referenced in provider configs.
            mcp_provider_registry: Registry of MCP provider configurations.
        """
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
    def tool_call_count(completion_response: Any) -> int:
        """Count the number of tool calls in a completion response.

        Args:
            completion_response: The LLM completion response (litellm.ModelResponse).

        Returns:
            Number of tool calls in the response (0 if none).
        """
        message = completion_response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None:
            return 0
        return len(tool_calls)

    @staticmethod
    def has_tool_calls(completion_response: Any) -> bool:
        """Returns True if tool calls are present in the completion response."""
        return MCPFacade.tool_call_count(completion_response) > 0

    def _resolve_provider(self, provider: MCPProviderT) -> MCPProviderT:
        """Resolve secret references in an MCP provider's api_key.

        Creates a copy of the provider with the api_key resolved from any secret
        reference (e.g., "env:API_KEY") to its actual value.

        Args:
            provider: The MCP provider config.

        Returns:
            A copy of the provider with resolved api_key, or the original provider
            if no api_key is configured.
        """
        api_key_ref = getattr(provider, "api_key", None)
        if not api_key_ref:
            return provider
        resolved_key = self._secret_resolver.resolve(api_key_ref)
        return provider.model_copy(update={"api_key": resolved_key})

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
        completion_response: Any,
    ) -> list[ChatMessage]:
        """Process an LLM completion response and execute any tool calls.

        This is the primary method for handling tool calls from an LLM response.
        It extracts the response content, reasoning content, and all tool calls
        from the completion response, executes each tool call (including parallel
        tool calls), and returns the messages for continuing the conversation.

        Args:
            completion_response: The completion response object from the LLM,
                typically from `router.completion()`. Expected to have a
                `choices[0].message` structure with optional `content`,
                `reasoning_content`, and `tool_calls` attributes.

        Returns:
            A list of ChatMessages to append to the conversation history:
            - If tool calls were present: [assistant_message_with_tool_calls, *tool_response_messages]
            - If no tool calls: [assistant_message]

        Raises:
            MCPToolError: If a tool call is missing a name.
            MCPToolError: If tool call arguments cannot be parsed as JSON.
            MCPToolError: If tool call arguments are an unsupported type.
            MCPToolError: If a requested tool is not in the allowed tools list.
            MCPToolError: If tool execution fails or times out.
            MCPConfigurationError: If a requested tool is not found on any configured provider.
        """
        message = completion_response.choices[0].message

        # Extract response content and reasoning content
        response_content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)

        # Strip whitespace if reasoning is present (models often add extra newlines)
        if reasoning_content:
            response_content = response_content.strip()
            reasoning_content = reasoning_content.strip()

        # Extract and normalize tool calls
        tool_calls = self._extract_tool_calls(message)

        if not tool_calls:
            # No tool calls - just return the assistant message
            return [
                ChatMessage.as_assistant(
                    content=response_content,
                    reasoning_content=reasoning_content or None,
                )
            ]

        # Has tool calls - execute and return all messages
        assistant_message = self._build_assistant_tool_message(response_content, tool_calls, reasoning_content)
        tool_messages = self._execute_tool_calls_internal(tool_calls)

        return [assistant_message, *tool_messages]

    def refuse_completion_response(
        self,
        completion_response: Any,
        refusal_message: str | None = None,
    ) -> list[ChatMessage]:
        """Refuse tool calls without executing them.

        Used when the tool call turn budget is exhausted. Returns messages
        that include the assistant's tool call request but with refusal
        responses instead of actual tool results. This allows the model
        to gracefully degrade and provide a final response without tools.

        Args:
            completion_response: The LLM completion response containing tool calls.
            refusal_message: Optional custom refusal message. Defaults to a
                standard message about tool budget exhaustion.

        Returns:
            A list of ChatMessages to append to the conversation history:
            - If tool calls were present: [assistant_message_with_tool_calls, *refusal_messages]
            - If no tool calls: [assistant_message]
        """
        message = completion_response.choices[0].message

        # Extract response content and reasoning content
        response_content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)

        # Strip whitespace if reasoning is present (models often add extra newlines)
        if reasoning_content:
            response_content = response_content.strip()
            reasoning_content = reasoning_content.strip()

        # Extract and normalize tool calls
        tool_calls = self._extract_tool_calls(message)

        if not tool_calls:
            # No tool calls to refuse - just return assistant message
            return [
                ChatMessage.as_assistant(
                    content=response_content,
                    reasoning_content=reasoning_content or None,
                )
            ]

        # Build assistant message with tool calls (same as normal)
        assistant_message = self._build_assistant_tool_message(response_content, tool_calls, reasoning_content)

        # Build refusal messages instead of executing tools
        refusal = refusal_message or DEFAULT_TOOL_REFUSAL_MESSAGE
        tool_messages = [ChatMessage.as_tool(content=refusal, tool_call_id=tc["id"]) for tc in tool_calls]

        return [assistant_message, *tool_messages]

    def _extract_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        """Extract and normalize tool calls from an LLM response message.

        Handles various LLM response formats (dict or object with attributes)
        and normalizes them into a consistent dictionary format. Supports
        parallel tool calling where the model returns multiple tool calls
        in a single response.

        Args:
            message: The LLM response message, either as a dictionary or an object
                with a 'tool_calls' attribute.

        Returns:
            A list of normalized tool call dictionaries. Each dictionary contains:
                - 'id': Unique identifier for the tool call (generated if not provided)
                - 'name': The name of the tool to call
                - 'arguments': Parsed arguments as a dictionary
                - 'arguments_json': Arguments serialized as a JSON string

            Returns an empty list if no tool calls are present in the message.

        Raises:
            MCPToolError: If a tool call is missing a name.
            MCPToolError: If tool call arguments cannot be parsed as JSON.
            MCPToolError: If tool call arguments are an unsupported type.
        """
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls is None and isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls")
        if not raw_tool_calls:
            return []

        tool_calls: list[dict[str, Any]] = []
        for raw_tool_call in raw_tool_calls:
            tool_calls.append(self._normalize_tool_call(raw_tool_call))
        return tool_calls

    def _normalize_tool_call(self, raw_tool_call: Any) -> dict[str, Any]:
        """Normalize a tool call from various LLM response formats.

        Handles both dictionary and object representations of tool calls,
        supporting the OpenAI format (with nested 'function' key) and
        flattened formats.

        Args:
            raw_tool_call: A tool call in any supported format.

        Returns:
            A normalized tool call dictionary with keys:
                - 'id': Tool call identifier (UUID generated if not provided)
                - 'name': The tool name
                - 'arguments': Parsed arguments dictionary
                - 'arguments_json': JSON string of arguments

        Raises:
            MCPToolError: If the tool call is missing a name or has invalid
                arguments that cannot be parsed as JSON.
        """
        if isinstance(raw_tool_call, dict):
            tool_call_id = raw_tool_call.get("id")
            function = raw_tool_call.get("function") or {}
            name = function.get("name") or raw_tool_call.get("name")
            arguments = function.get("arguments") or raw_tool_call.get("arguments")
        else:
            tool_call_id = getattr(raw_tool_call, "id", None)
            function = getattr(raw_tool_call, "function", None)
            name = getattr(function, "name", None) if function is not None else getattr(raw_tool_call, "name", None)
            arguments = (
                getattr(function, "arguments", None)
                if function is not None
                else getattr(raw_tool_call, "arguments", None)
            )

        if not name:
            raise MCPToolError("MCP tool call is missing a tool name.")

        arguments_payload: dict[str, Any]
        arguments_json: str
        if arguments is None or arguments == "":
            arguments_payload = {}
            arguments_json = "{}"
        elif isinstance(arguments, str):
            try:
                arguments_payload = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise MCPToolError(f"Invalid tool arguments for '{name}': {arguments}") from exc
            arguments_json = arguments
        elif isinstance(arguments, dict):
            arguments_payload = arguments
            arguments_json = json.dumps(arguments_payload)
        else:
            raise MCPToolError(f"Unsupported tool arguments type for '{name}': {type(arguments)!r}")

        return {
            "id": tool_call_id or uuid.uuid4().hex,
            "name": name,
            "arguments": arguments_payload,
            "arguments_json": arguments_json,
        }

    def _build_assistant_tool_message(
        self,
        response: str | None,
        tool_calls: list[dict[str, Any]],
        reasoning_content: str | None = None,
    ) -> ChatMessage:
        """Build the assistant message containing tool call requests.

        Constructs a message in the format expected by the LLM conversation
        history, representing the assistant's request to call tools.

        Args:
            response: The assistant's text response content. May be empty if
                the assistant only requested tool calls without additional text.
            tool_calls: List of normalized tool call dictionaries.
            reasoning_content: Optional reasoning content from the assistant's
                response. If provided, will be included under the 'reasoning_content' key.

        Returns:
            A ChatMessage representing the assistant message with tool call requests.
        """
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

    def _execute_tool_calls_internal(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ChatMessage]:
        """Execute tool calls in parallel and return tool response messages.

        Validates all tool calls, then executes them concurrently via the io module
        using call_tools_parallel. This leverages parallel tool calling when the
        model returns multiple tool calls in a single response.

        Args:
            tool_calls: List of normalized tool call dictionaries to execute.

        Returns:
            A list of tool response messages, one per tool call.

        Raises:
            MCPToolError: If a tool is not in the allowed tools list or if
                the MCP provider returns an error.
        """
        allowed_tools = set(self._tool_config.allow_tools) if self._tool_config.allow_tools else None

        # Validate all tool calls and collect provider + args
        calls_to_execute: list[tuple[MCPProviderT, str, dict[str, Any], str]] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            if allowed_tools is not None and tool_name not in allowed_tools:
                providers_str = ", ".join(repr(p) for p in self._tool_config.providers)
                raise MCPToolError(f"Tool {tool_name!r} is not permitted for providers: {providers_str}.")

            resolved_provider = self._find_resolved_provider_for_tool(tool_name)
            calls_to_execute.append((resolved_provider, tool_name, tool_call["arguments"], tool_call["id"]))

        # Execute all calls in parallel
        results = mcp_io.call_tools(
            [(p, n, a) for p, n, a, _ in calls_to_execute],
            timeout_sec=self._tool_config.timeout_sec,
        )

        # Build response messages
        return [
            ChatMessage.as_tool(content=result.content, tool_call_id=call[3])
            for result, call in zip(results, calls_to_execute)
        ]

    def _find_resolved_provider_for_tool(self, tool_name: str) -> MCPProviderT:
        """Find the provider that has the given tool and return it with resolved api_key.

        Args:
            tool_name: The name of the tool to find.

        Returns:
            The provider object (with resolved api_key) that has the tool.

        Raises:
            MCPConfigurationError: If no provider has the tool.
        """
        for provider_name in self._tool_config.providers:
            provider = self._mcp_provider_registry.get_provider(provider_name)
            resolved_provider = self._resolve_provider(provider)
            tools = mcp_io.list_tools(
                resolved_provider, timeout_sec=self._tool_config.timeout_sec
            )  # Cached in io module
            if any(tool.name == tool_name for tool in tools):
                return resolved_provider

        raise MCPConfigurationError(f"Tool {tool_name!r} not found on any configured provider.")
