# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from data_designer.config.mcp import MCPServerConfig, MCPToolConfig
from data_designer.engine.mcp.errors import MCPClientUnavailableError, MCPConfigurationError, MCPToolError


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    description: str | None
    input_schema: dict[str, Any] | None


@dataclass(frozen=True)
class MCPToolResult:
    content: str
    is_error: bool = False


class MCPClientManager:
    def __init__(self, *, server_configs: list[MCPServerConfig]):
        self._server_configs = self._build_server_map(server_configs)
        self._tool_cache: dict[str, list[MCPToolDefinition]] = {}
        self._async_executor = ThreadPoolExecutor(max_workers=1)

    def get_tool_schemas(self, tool_config: MCPToolConfig) -> list[dict[str, Any]]:
        server = self._get_server(tool_config.server_name)
        tools = self._list_tools(server.name)
        allowed_names = set(tool_config.tool_names) if tool_config.tool_names else None
        if allowed_names is not None:
            available = {tool.name for tool in tools}
            missing = allowed_names.difference(available)
            if missing:
                raise MCPConfigurationError(f"Tool(s) {sorted(missing)!r} not found on MCP server {server.name!r}.")
            tools = [tool for tool in tools if tool.name in allowed_names]
        return [self._to_openai_tool_schema(tool) for tool in tools]

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        server = self._get_server(server_name)
        result = self._run_async(self._call_tool_async(server, tool_name, arguments))
        return result

    def _build_server_map(self, server_configs: list[MCPServerConfig]) -> dict[str, MCPServerConfig]:
        server_map: dict[str, MCPServerConfig] = {}
        for config in server_configs:
            if config.name in server_map:
                raise MCPConfigurationError(f"Duplicate MCP server name {config.name!r} detected.")
            server_map[config.name] = config
        return server_map

    def _get_server(self, name: str) -> MCPServerConfig:
        try:
            return self._server_configs[name]
        except KeyError as exc:
            raise MCPConfigurationError(f"No MCP server named {name!r} is configured.") from exc

    def _list_tools(self, server_name: str) -> list[MCPToolDefinition]:
        if server_name in self._tool_cache:
            return self._tool_cache[server_name]
        server = self._get_server(server_name)
        tools = self._run_async(self._list_tools_async(server))
        self._tool_cache[server_name] = tools
        return tools

    def _run_async(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        future = self._async_executor.submit(asyncio.run, coro)
        return future.result()

    async def _list_tools_async(self, server: MCPServerConfig) -> list[MCPToolDefinition]:
        ClientSession, StdioServerParameters, stdio_client, sse_client = _resolve_mcp_imports()
        if server.command:
            params = StdioServerParameters(command=server.command, args=server.args, env=server.env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
        else:
            async with sse_client(server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()

        raw_tools = getattr(result, "tools", result)
        if not isinstance(raw_tools, list):
            raise MCPToolError("Unexpected response from MCP server when listing tools.")
        return [self._coerce_tool_definition(tool) for tool in raw_tools]

    async def _call_tool_async(
        self, server: MCPServerConfig, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        ClientSession, StdioServerParameters, stdio_client, sse_client = _resolve_mcp_imports()
        if server.command:
            params = StdioServerParameters(command=server.command, args=server.args, env=server.env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
        else:
            async with sse_client(server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)

        content = _serialize_tool_result_content(result)
        is_error = getattr(result, "isError", None)
        if is_error is None:
            is_error = getattr(result, "is_error", False)
        return MCPToolResult(content=content, is_error=bool(is_error))

    def _coerce_tool_definition(self, tool: Any) -> MCPToolDefinition:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description")
            input_schema = tool.get("inputSchema") or tool.get("input_schema")
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)

        if not name:
            raise MCPToolError("Encountered MCP tool without a name.")
        return MCPToolDefinition(name=name, description=description, input_schema=input_schema)

    def _to_openai_tool_schema(self, tool: MCPToolDefinition) -> dict[str, Any]:
        schema = tool.input_schema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        }


def _serialize_tool_result_content(result: Any) -> str:
    content = getattr(result, "content", result)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item))
                continue
            text_value = getattr(item, "text", None)
            if text_value is not None:
                parts.append(str(text_value))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _resolve_mcp_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client

        return ClientSession, StdioServerParameters, stdio_client, sse_client
    except ImportError:
        try:
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client
            from mcp.client.stdio import StdioServerParameters, stdio_client

            return ClientSession, StdioServerParameters, stdio_client, sse_client
        except ImportError as exc:
            raise MCPClientUnavailableError(
                "MCP client dependencies are not installed. Install the 'mcp' package to enable tool calling."
            ) from exc
