# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from data_designer.config.mcp import MCPToolConfig
from data_designer.engine.mcp.errors import MCPConfigurationError, MCPToolError
from data_designer.engine.models.utils import ChatMessage

if TYPE_CHECKING:
    from data_designer.engine.mcp.manager import MCPClientManager


@dataclass
class ToolExecutionResult:
    """Result of executing tool calls within an LLM generation loop.

    This dataclass encapsulates all the outputs from executing one or more
    MCP tool calls, providing the messages needed to continue the conversation
    with the LLM.

    Attributes:
        assistant_message: The assistant message containing the tool call requests.
            This message should be appended to the conversation history before
            the tool response messages.
        tool_messages: List of tool response messages, one per executed tool call.
        tool_calls_count: The number of tool calls that were executed.
    """

    assistant_message: ChatMessage
    tool_messages: list[ChatMessage]
    tool_calls_count: int


class MCPToolExecutor:
    """Handles MCP tool call extraction, normalization, and execution.

    This class extracts tool-related logic from the ModelFacade to keep
    the facade focused on LLM generation while delegating tool execution
    to a dedicated component. It provides methods for:

    - Retrieving tool schemas in OpenAI-compatible format
    - Extracting and normalizing tool calls from LLM responses
    - Executing tool calls via the MCP client manager

    Attributes:
        _mcp_client_manager: The MCP client manager for communicating with MCP servers.
            May be None if MCP is not configured.
    """

    def __init__(self, mcp_client_manager: MCPClientManager | None = None) -> None:
        """Initialize the MCPToolExecutor.

        Args:
            mcp_client_manager: The MCP client manager instance for communicating
                with configured MCP servers. If None, tool operations will raise
                MCPConfigurationError when attempted.
        """
        self._mcp_client_manager = mcp_client_manager

    def get_tool_schemas(self, tool_config: MCPToolConfig) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for the given tool configuration.

        Retrieves the available tools from the configured MCP server and converts
        them to the OpenAI function calling format expected by LiteLLM.

        Args:
            tool_config: The MCP tool configuration specifying which server to use
                and optionally which tools to include.

        Returns:
            A list of tool schema dictionaries in OpenAI function calling format.
            Each schema contains 'type' (always 'function') and 'function' keys,
            where 'function' includes 'name', 'description', and 'parameters'.

        Raises:
            MCPConfigurationError: If no MCP client manager was configured.
        """
        if self._mcp_client_manager is None:
            raise MCPConfigurationError("MCP tool configuration was provided but no MCP servers were configured.")
        return self._mcp_client_manager.get_tool_schemas(tool_config)

    def process_completion_response(
        self, completion_response: Any, tool_config: MCPToolConfig
    ) -> ToolExecutionResult | None:
        """Process an LLM completion response and execute any tool calls.

        This is the primary method for handling tool calls from an LLM response.
        It extracts the response content, reasoning content, and all tool calls
        from the completion response, executes each tool call (including parallel
        tool calls), and returns the results packaged for continuing the conversation.

        Args:
            completion_response: The completion response object from the LLM,
                typically from `router.completion()`. Expected to have a
                `choices[0].message` structure with optional `content`,
                `reasoning_content`, and `tool_calls` attributes.
            tool_config: The MCP tool configuration specifying the server,
                allowed tools, and timeout settings.

        Returns:
            A ToolExecutionResult if tool calls were present and executed,
            containing:
                - assistant_message: The assistant message with tool call requests
                  and optional reasoning content
                - tool_messages: List of tool response messages (one per tool call)
                - tool_calls_count: Number of tools that were called

            Returns None if no tool calls were present in the response.

        Raises:
            MCPConfigurationError: If no MCP client manager was configured.
            MCPToolError: If a requested tool is not in the allowed tools list
                or if tool execution fails.
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
            return None

        # Execute all tool calls (handles parallel tool calling)
        return self._execute_tool_calls(tool_config, tool_calls, response_content, reasoning_content)

    def _extract_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        """Extract and normalize tool calls from an LLM response message.

        Handles various LLM response formats (dict or object with attributes)
        and normalizes them into a consistent dictionary format. Supports
        parallel tool calling where the model returns multiple tool calls
        in a single response.

        Args:
            message: The LLM response message, either as a dictionary or an object
                with a 'tool_calls' attribute. Typically this is the message object
                from `completion_response.choices[0].message`.

        Returns:
            A list of normalized tool call dictionaries. Each dictionary contains:
                - 'id': Unique identifier for the tool call (generated if not provided)
                - 'name': The name of the tool to call
                - 'arguments': Parsed arguments as a dictionary
                - 'arguments_json': Arguments serialized as a JSON string

            Returns an empty list if no tool calls are present in the message.
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

    def _execute_tool_calls(
        self,
        tool_config: MCPToolConfig,
        tool_calls: list[dict[str, Any]],
        response_content: str,
        reasoning_content: str | None = None,
    ) -> ToolExecutionResult:
        """Execute tool calls and return messages to append to the conversation.

        This method executes each tool call against the configured MCP server
        and packages the results into messages suitable for continuing the
        LLM conversation.

        Args:
            tool_config: The MCP tool configuration specifying the server,
                allowed tools, and timeout settings.
            tool_calls: List of normalized tool calls to execute, as returned
                by `extract_tool_calls()`.
            response_content: The assistant's response content that accompanied
                the tool calls. May be empty string if the LLM only returned
                tool calls without additional text.
            reasoning_content: Optional reasoning content from the assistant's
                response (e.g., from extended thinking / chain-of-thought models).
                If provided, will be included in the assistant message.

        Returns:
            A ToolExecutionResult containing:
                - assistant_message: The assistant message with tool call requests
                  (includes reasoning_content if provided)
                - tool_messages: List of tool response messages
                - tool_calls_count: Number of tools that were called

        Raises:
            MCPConfigurationError: If no MCP client manager was configured.
            MCPToolError: If a requested tool is not in the allowed tools list
                or if tool execution fails.
        """
        if self._mcp_client_manager is None:
            raise MCPConfigurationError("MCP tool configuration was provided but no MCP servers were configured.")

        assistant_message = self._build_assistant_tool_message(response_content, tool_calls, reasoning_content)
        tool_messages = self._execute_tool_calls_internal(tool_config, tool_calls)

        return ToolExecutionResult(
            assistant_message=assistant_message,
            tool_messages=tool_messages,
            tool_calls_count=len(tool_calls),
        )

    def _normalize_tool_call(self, raw_tool_call: Any) -> dict[str, Any]:
        """Normalize a tool call from various LLM response formats.

        Handles both dictionary and object representations of tool calls,
        supporting the OpenAI format (with nested 'function' key) and
        flattened formats.

        Args:
            raw_tool_call: A tool call in any supported format. Can be a dictionary
                with 'id', 'function.name', 'function.arguments' keys, or an object
                with corresponding attributes.

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
        self, response: str | None, tool_calls: list[dict[str, Any]], reasoning_content: str | None = None
    ) -> ChatMessage:
        """Build the assistant message containing tool call requests.

        Constructs a message in the format expected by the LLM conversation
        history, representing the assistant's request to call tools.

        Args:
            response: The assistant's text response content. May be empty if
                the assistant only requested tool calls without additional text.
            tool_calls: List of normalized tool call dictionaries as returned
                by `_normalize_tool_call()`.
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
        return ChatMessage.assistant(
            content=response or "",
            reasoning_content=reasoning_content or None,
            tool_calls=tool_calls_payload,
        )

    def _execute_tool_calls_internal(
        self, tool_config: MCPToolConfig, tool_calls: list[dict[str, Any]]
    ) -> list[ChatMessage]:
        """Execute tool calls and return tool response messages.

        Iterates through the tool calls, validates each against the allowed
        tools list, executes them via the MCP client manager, and collects
        the responses.

        Args:
            tool_config: The MCP tool configuration specifying the server name,
                allowed tools, and timeout settings.
            tool_calls: List of normalized tool call dictionaries to execute.

        Returns:
            A list of tool response messages, one per tool call.

        Raises:
            MCPConfigurationError: If no MCP client manager was configured.
            MCPToolError: If a tool is not in the allowed tools list or if
                the MCP server returns an error.
        """
        if self._mcp_client_manager is None:
            raise MCPConfigurationError("MCP tool configuration was provided but no MCP servers were configured.")

        allowed_tools = set(tool_config.tool_names) if tool_config.tool_names else None
        tool_messages: list[ChatMessage] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            if allowed_tools is not None and tool_name not in allowed_tools:
                raise MCPToolError(f"Tool {tool_name!r} is not permitted for server {tool_config.server_name!r}.")
            result = self._mcp_client_manager.call_tool(
                tool_config.server_name,
                tool_name,
                tool_call["arguments"],
                timeout_sec=tool_config.timeout_sec,
            )
            tool_messages.append(ChatMessage.tool(content=result.content, tool_call_id=tool_call["id"]))
        return tool_messages
