# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level MCP I/O operations with caching and session pooling.

This module provides stateless functions for MCP communication using an actor-style
service that owns all async state within a single background event loop. Public APIs
are synchronous wrappers that submit coroutines to the loop and wait for results.

Architecture:
    All MCP I/O is funneled through a single dedicated asyncio event loop running
    in a background daemon thread. This avoids the complexity of managing multiple
    event loops and allows sessions to be reused across calls from any thread.

    Worker Thread 1 ──┐
    Worker Thread 2 ──┼──► MCP Event Loop Thread ──► MCP Servers
    Worker Thread N ──┘    (all sessions live here)

Request Coalescing:
    When multiple threads request tools from the same provider simultaneously,
    only one request is made to the MCP server. Other callers wait for the
    in-flight request to complete and share the result. This prevents N
    concurrent workers from making N separate ListToolsRequest calls.

The caller (MCPFacade) is responsible for resolving any secret references in
provider api_key fields before passing providers to these functions.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import threading
from collections.abc import Coroutine, Iterable
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, MCPProviderT
from data_designer.engine.mcp.errors import MCPToolError
from data_designer.engine.mcp.registry import MCPToolDefinition, MCPToolResult

logger = logging.getLogger(__name__)


def _provider_cache_key(provider: MCPProviderT) -> str:
    """Create a stable cache key for a provider."""
    data = provider.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class MCPIOService:
    """Actor-style MCP I/O service owning all async state."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()

        self._sessions: dict[str, ClientSession] = {}
        self._session_contexts: dict[str, Any] = {}
        self._session_inflight: dict[str, asyncio.Task[ClientSession]] = {}

        self._tools_cache: dict[str, tuple[MCPToolDefinition, ...]] = {}
        self._tools_cache_epoch: dict[str, int] = {}
        self._inflight_tools: dict[str, asyncio.Task[tuple[MCPToolDefinition, ...]]] = {}

    def list_tools(self, provider: MCPProviderT, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        """List tools from an MCP provider (cached with request coalescing)."""
        try:
            return self._run_on_loop(self._list_tools_async(provider), timeout_sec)
        except TimeoutError as exc:
            timeout_label = f"{timeout_sec:.1f}" if timeout_sec is not None else "unknown"
            raise MCPToolError(f"Timed out after {timeout_label}s while listing tools on {provider.name!r}.") from exc

    def call_tools(
        self,
        calls: list[tuple[MCPProviderT, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        """Call multiple tools in parallel."""
        if not calls:
            return []
        try:
            return self._run_on_loop(self._call_tools_async(calls), timeout_sec)
        except TimeoutError as exc:
            timeout_label = f"{timeout_sec:.1f}" if timeout_sec is not None else "unknown"
            raise MCPToolError(f"Timed out after {timeout_label}s while calling tools in parallel.") from exc

    def clear_provider_caches(self, providers: list[MCPProviderT]) -> int:
        """Clear caches and session pool entries for specific providers."""
        if not providers:
            return 0
        if self._loop is not None and self._loop.is_running():
            try:
                return self._run_on_loop(self._clear_provider_caches_async(providers), timeout_sec=5)
            except Exception:
                logger.debug("Failed to clear provider caches on MCP IO service.", exc_info=True)
                return 0
        return self._clear_provider_caches_sync(providers)

    def clear_tools_cache(self) -> None:
        """Clear the list_tools cache (best effort)."""
        if self._loop is not None and self._loop.is_running():
            try:
                self._run_on_loop(self._clear_tools_cache_async(), timeout_sec=5)
                return
            except Exception:
                logger.debug("Failed to clear tools cache on MCP IO service.", exc_info=True)
                return
        self._clear_tools_cache_sync()

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics for list_tools."""
        if self._loop is not None and self._loop.is_running():
            try:
                return self._run_on_loop(self._get_cache_info_async(), timeout_sec=5)
            except Exception:
                logger.debug("Failed to read tools cache info on MCP IO service.", exc_info=True)
        return {"currsize": len(self._tools_cache), "providers": list(self._tools_cache.keys())}

    def clear_session_pool(self) -> None:
        """Clear all pooled MCP sessions (best effort)."""
        if self._loop is not None and self._loop.is_running():
            try:
                self._run_on_loop(self._close_all_sessions_async(), timeout_sec=5)
                return
            except Exception:
                logger.debug("Failed to clear session pool on MCP IO service.", exc_info=True)
                # Fall through to sync cleanup
        self._clear_session_pool_sync()

    def get_session_pool_info(self) -> dict[str, Any]:
        """Get information about the session pool."""
        if self._loop is not None and self._loop.is_running():
            try:
                return self._run_on_loop(self._get_session_pool_info_async(), timeout_sec=5)
            except Exception:
                logger.debug("Failed to read session pool info on MCP IO service.", exc_info=True)
        return {"active_sessions": len(self._sessions), "provider_keys": list(self._sessions.keys())}

    def shutdown(self) -> None:
        """Shutdown the MCP event loop and close all sessions."""
        if self._loop is None:
            self._reset_state()
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._close_all_sessions_async(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
        finally:
            self._loop = None
            self._thread = None
            self._reset_state()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None or not self._loop.is_running():
                loop = asyncio.new_event_loop()
                self._loop = loop
                self._thread = threading.Thread(
                    target=self._run_loop,
                    args=(loop,),
                    daemon=True,
                    name="MCP-EventLoop",
                )
                self._thread.start()
                logger.debug("Started MCP background event loop")
            # Capture local reference to avoid race with concurrent shutdown()
            loop = self._loop
        return loop

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _run_on_loop(self, coro: Coroutine[Any, Any, Any], timeout_sec: float | None) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout_sec)

    async def _get_or_create_session(self, provider: MCPProviderT) -> ClientSession:
        key = _provider_cache_key(provider)
        session = self._sessions.get(key)
        if session is not None:
            return session

        inflight = self._session_inflight.get(key)
        if inflight is not None:
            return await inflight

        async def create_session() -> ClientSession:
            ctx: Any | None = None
            new_session: ClientSession | None = None
            try:
                if isinstance(provider, LocalStdioMCPProvider):
                    params = StdioServerParameters(
                        command=provider.command,
                        args=provider.args,
                        env=provider.env,
                    )
                    ctx = stdio_client(params)
                elif isinstance(provider, MCPProvider) and provider.provider_type == "streamable_http":
                    headers = _build_auth_headers(provider.api_key)
                    ctx = streamablehttp_client(provider.endpoint, headers=headers)
                else:
                    headers = _build_auth_headers(provider.api_key)
                    ctx = sse_client(provider.endpoint, headers=headers)

                ctx_result = await ctx.__aenter__()
                read, write = ctx_result[0], ctx_result[1]
                new_session = ClientSession(read, write)
                await new_session.__aenter__()
                await new_session.initialize()

                self._sessions[key] = new_session
                self._session_contexts[key] = ctx
                logger.debug("Created pooled MCP session for provider %r", provider.name)
                return new_session
            except Exception:
                if new_session is not None:
                    try:
                        await new_session.__aexit__(None, None, None)
                    except Exception:
                        pass
                if ctx is not None:
                    try:
                        await ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
                raise

        task = asyncio.create_task(create_session())
        self._session_inflight[key] = task
        try:
            return await task
        finally:
            self._session_inflight.pop(key, None)

    async def _list_tools_async(self, provider: MCPProviderT) -> tuple[MCPToolDefinition, ...]:
        key = _provider_cache_key(provider)
        cached = self._tools_cache.get(key)
        if cached is not None:
            return cached

        inflight = self._inflight_tools.get(key)
        if inflight is not None:
            return await inflight

        epoch = self._tools_cache_epoch.get(key, 0)

        async def fetch_tools() -> tuple[MCPToolDefinition, ...]:
            session = await self._get_or_create_session(provider)
            result = await session.list_tools()
            raw_tools = getattr(result, "tools", result)
            if not isinstance(raw_tools, list):
                raise MCPToolError("Unexpected response from MCP provider when listing tools.")
            tools = tuple(_coerce_tool_definition(tool, MCPToolDefinition) for tool in raw_tools)
            if self._tools_cache_epoch.get(key, 0) == epoch:
                self._tools_cache[key] = tools
                logger.debug("Cached tools for provider %r (%d tools)", provider.name, len(tools))
            return tools

        task = asyncio.create_task(fetch_tools())
        self._inflight_tools[key] = task
        try:
            return await task
        finally:
            self._inflight_tools.pop(key, None)

    async def _call_tool_async(
        self,
        provider: MCPProviderT,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        session = await self._get_or_create_session(provider)
        result = await session.call_tool(tool_name, arguments)

        content = _serialize_tool_result_content(result)
        is_error = getattr(result, "isError", None)
        if is_error is None:
            is_error = getattr(result, "is_error", False)

        return MCPToolResult(content=content, is_error=bool(is_error))

    async def _call_tools_async(
        self,
        calls: list[tuple[MCPProviderT, str, dict[str, Any]]],
    ) -> list[MCPToolResult]:
        return await asyncio.gather(*[self._call_tool_async(p, n, a) for p, n, a in calls])

    async def _clear_provider_caches_async(self, providers: list[MCPProviderT]) -> int:
        keys = [_provider_cache_key(provider) for provider in providers]
        self._invalidate_tools_cache(keys)

        cleared_count = 0
        for key in keys:
            session = self._sessions.pop(key, None)
            ctx = self._session_contexts.pop(key, None)
            if session is not None:
                cleared_count += 1
                try:
                    await session.__aexit__(None, None, None)
                except Exception:
                    pass
            if ctx is not None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass

        if cleared_count > 0:
            logger.debug("Cleared %d provider cache entries", cleared_count)
        return cleared_count

    def _clear_provider_caches_sync(self, providers: list[MCPProviderT]) -> int:
        keys = [_provider_cache_key(provider) for provider in providers]
        self._invalidate_tools_cache(keys)

        cleared_count = 0
        for key in keys:
            if key in self._sessions:
                del self._sessions[key]
                cleared_count += 1
            if key in self._session_contexts:
                del self._session_contexts[key]

        if cleared_count > 0:
            logger.debug("Cleared %d provider cache entries", cleared_count)
        return cleared_count

    async def _clear_tools_cache_async(self) -> None:
        self._invalidate_tools_cache(self._all_tools_keys())

    def _clear_tools_cache_sync(self) -> None:
        self._invalidate_tools_cache(self._all_tools_keys())

    async def _get_cache_info_async(self) -> dict[str, Any]:
        return {"currsize": len(self._tools_cache), "providers": list(self._tools_cache.keys())}

    async def _close_all_sessions_async(self) -> None:
        for key in list(self._sessions.keys()):
            session = self._sessions.pop(key, None)
            ctx = self._session_contexts.pop(key, None)
            if session is not None:
                try:
                    await session.__aexit__(None, None, None)
                except Exception:
                    pass
            if ctx is not None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass

        for task in self._session_inflight.values():
            task.cancel()
        self._session_inflight.clear()

    def _clear_session_pool_sync(self) -> None:
        self._sessions.clear()
        self._session_contexts.clear()
        self._session_inflight.clear()

    async def _get_session_pool_info_async(self) -> dict[str, Any]:
        return {"active_sessions": len(self._sessions), "provider_keys": list(self._sessions.keys())}

    def _invalidate_tools_cache(self, keys: Iterable[str]) -> None:
        for key in keys:
            self._tools_cache.pop(key, None)
            self._tools_cache_epoch[key] = self._tools_cache_epoch.get(key, 0) + 1

    def _all_tools_keys(self) -> set[str]:
        return set(self._tools_cache) | set(self._inflight_tools) | set(self._tools_cache_epoch)

    def _reset_state(self) -> None:
        self._sessions.clear()
        self._session_contexts.clear()
        self._session_inflight.clear()
        self._tools_cache.clear()
        self._tools_cache_epoch.clear()
        self._inflight_tools.clear()


_MCP_IO_SERVICE = MCPIOService()
atexit.register(_MCP_IO_SERVICE.shutdown)


def list_tools(provider: MCPProviderT, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
    """List tools from an MCP provider (cached with request coalescing)."""
    return _MCP_IO_SERVICE.list_tools(provider, timeout_sec=timeout_sec)


def list_tool_names(provider: MCPProviderT, timeout_sec: float) -> list[str]:
    """Return the names of all tools available on an MCP provider."""
    return [t.name for t in _MCP_IO_SERVICE.list_tools(provider, timeout_sec=timeout_sec)]


def call_tools(
    calls: list[tuple[MCPProviderT, str, dict[str, Any]]],
    *,
    timeout_sec: float | None = None,
) -> list[MCPToolResult]:
    """Call multiple tools in parallel."""
    return _MCP_IO_SERVICE.call_tools(calls, timeout_sec=timeout_sec)


def clear_provider_caches(providers: list[MCPProviderT]) -> int:
    """Clear all caches for specific MCP providers."""
    return _MCP_IO_SERVICE.clear_provider_caches(providers)


def clear_tools_cache() -> None:
    """Clear the list_tools cache."""
    _MCP_IO_SERVICE.clear_tools_cache()


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics for list_tools."""
    return _MCP_IO_SERVICE.get_cache_info()


def clear_session_pool() -> None:
    """Clear all pooled MCP sessions."""
    _MCP_IO_SERVICE.clear_session_pool()


def get_session_pool_info() -> dict[str, Any]:
    """Get information about the session pool."""
    return _MCP_IO_SERVICE.get_session_pool_info()


def _build_auth_headers(api_key: str | None) -> dict[str, Any] | None:
    """Build authentication headers for remote MCP clients."""
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def _coerce_tool_definition(tool: Any, tool_definition_cls: type[MCPToolDefinition]) -> MCPToolDefinition:
    """Coerce a tool from various formats into MCPToolDefinition."""
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

    return tool_definition_cls(name=name, description=description, input_schema=input_schema)


def _serialize_tool_result_content(result: Any) -> str:
    """Serialize tool result content to a string."""
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
