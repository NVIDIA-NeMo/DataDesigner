# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from data_designer.config.mcp import MCPServerConfig, MCPToolConfig
from data_designer.engine.mcp.errors import MCPConfigurationError, MCPToolError

logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        *,
        server_configs: list[MCPServerConfig],
        max_async_workers: int | None = None,
    ):
        self._server_configs = self._build_server_map(server_configs)
        self._tool_cache: dict[str, list[MCPToolDefinition]] = {}
        self._tool_cache_lock = Lock()
        self._async_executor = ThreadPoolExecutor(
            max_workers=self._resolve_async_workers(max_async_workers),
            thread_name_prefix="MCPClientManager",
        )

    @staticmethod
    def _resolve_async_workers(max_async_workers: int | None) -> int:
        if max_async_workers is not None:
            return max(1, max_async_workers)
        env_value = os.environ.get("DATA_DESIGNER_MCP_ASYNC_WORKERS")
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                pass
        cpu_count = os.cpu_count() or 4
        return max(4, min(32, cpu_count))

    def get_tool_schemas(self, tool_config: MCPToolConfig) -> list[dict[str, Any]]:
        server = self._get_server(tool_config.server_name)
        tools = self._list_tools(server.name, timeout_sec=tool_config.timeout_sec)
        allowed_names = set(tool_config.tool_names) if tool_config.tool_names else None
        if allowed_names is not None:
            available = {tool.name for tool in tools}
            missing = allowed_names.difference(available)
            if missing:
                raise MCPConfigurationError(f"Tool(s) {sorted(missing)!r} not found on MCP server {server.name!r}.")
            tools = [tool for tool in tools if tool.name in allowed_names]
        return [self._to_openai_tool_schema(tool) for tool in tools]

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout_sec: float | None = None,
    ) -> MCPToolResult:
        server = self._get_server(server_name)
        start_time = time.monotonic()
        result = self._run_async(
            self._call_tool_async(server, tool_name, arguments),
            operation=f"calling tool {tool_name!r} on {server.name!r}",
            timeout_sec=timeout_sec,
        )
        elapsed = time.monotonic() - start_time
        logger.debug("MCP tool %s on %s completed in %.2fs.", tool_name, server.name, elapsed)
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

    def _list_tools(self, server_name: str, *, timeout_sec: float | None = None) -> list[MCPToolDefinition]:
        if server_name in self._tool_cache:
            return self._tool_cache[server_name]
        with self._tool_cache_lock:
            if server_name in self._tool_cache:
                return self._tool_cache[server_name]
            server = self._get_server(server_name)
            start_time = time.monotonic()
            tools = self._run_async(
                self._list_tools_async(server),
                operation=f"listing tools on {server.name!r}",
                timeout_sec=timeout_sec,
            )
            elapsed = time.monotonic() - start_time
            logger.debug("MCP tool list for %s completed in %.2fs.", server.name, elapsed)
            self._tool_cache[server_name] = tools
            return tools

    def _run_async(self, coro: Any, *, operation: str, timeout_sec: float | None = None) -> Any:
        if timeout_sec is not None:
            coro = asyncio.wait_for(coro, timeout=timeout_sec)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(coro)
            except TimeoutError as exc:
                timeout_label = f"{timeout_sec:.1f}" if timeout_sec is not None else "unknown"
                raise MCPToolError(f"Timed out after {timeout_label}s while {operation}.") from exc
        future = self._async_executor.submit(asyncio.run, coro)
        try:
            return future.result()
        except TimeoutError as exc:
            timeout_label = f"{timeout_sec:.1f}" if timeout_sec is not None else "unknown"
            raise MCPToolError(f"Timed out after {timeout_label}s while {operation}.") from exc

    async def _list_tools_async(self, server: MCPServerConfig) -> list[MCPToolDefinition]:
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
