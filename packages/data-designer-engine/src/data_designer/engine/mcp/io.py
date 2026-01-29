# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level MCP I/O operations with caching and session pooling.

This module provides stateless functions for MCP communication. The `list_tools`
function uses request coalescing to prevent thundering herd problems when many
concurrent workers start simultaneously.

Session pooling is implemented to reuse MCP connections across multiple tool calls,
avoiding the overhead of creating new connections and performing MCP handshakes
for every single call.

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
from typing import TYPE_CHECKING, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProviderT
from data_designer.engine.mcp.errors import MCPToolError

if TYPE_CHECKING:
    from data_designer.engine.mcp.registry import MCPToolDefinition, MCPToolResult

logger = logging.getLogger(__name__)

# =============================================================================
# Background Event Loop
# =============================================================================

_mcp_loop: asyncio.AbstractEventLoop | None = None
_mcp_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _provider_cache_key(provider: MCPProviderT) -> str:
    """Create a stable cache key for a provider.

    We intentionally derive cache/session keys from provider configuration, but
    must ensure the serialization is stable across logically identical provider
    instances (e.g., dict insertion order for nested structures like `env`).
    """
    data = provider.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Ensure the background MCP event loop is running.

    Lazily starts the background event loop thread on first use.
    Thread-safe for concurrent initialization.

    Returns:
        The background event loop.
    """
    global _mcp_loop, _mcp_thread

    with _loop_lock:
        if _mcp_loop is None or not _mcp_loop.is_running():
            _mcp_loop = asyncio.new_event_loop()
            _mcp_thread = threading.Thread(
                target=_mcp_loop.run_forever,
                daemon=True,
                name="MCP-EventLoop",
            )
            _mcp_thread.start()
            logger.debug("Started MCP background event loop")

    return _mcp_loop


# =============================================================================
# Session Pooling (runs in the background event loop)
# =============================================================================

# Session cache: keyed by provider's serialized JSON config
_sessions: dict[str, ClientSession] = {}

# Context managers for sessions (we need to keep them alive)
_session_contexts: dict[str, Any] = {}

# Lock for thread-safe access to session cache (only used from event loop thread)
_session_lock = asyncio.Lock()

# =============================================================================
# Tools Cache with Request Coalescing
# =============================================================================

# Tools cache: keyed by provider's serialized JSON config
_tools_cache: dict[str, tuple[MCPToolDefinition, ...]] = {}

# Per-key locks for request coalescing (prevents thundering herd)
_tools_locks: dict[str, asyncio.Lock] = {}

# Lock for creating new per-key locks
_tools_locks_lock = asyncio.Lock()


async def _get_or_create_session(provider: MCPProviderT) -> ClientSession:
    """Get an existing session from pool or create a new one.

    Sessions are cached by provider config, so the same provider will
    always return the same session (connection reuse).

    This function must be called from within the background event loop.

    Args:
        provider: The MCP provider config.

    Returns:
        A ClientSession connected to the provider.
    """
    key = _provider_cache_key(provider)

    async with _session_lock:
        if key in _sessions:
            return _sessions[key]

        # Create new session
        if isinstance(provider, LocalStdioMCPProvider):
            params = StdioServerParameters(
                command=provider.command,
                args=provider.args,
                env=provider.env,
            )
            ctx = stdio_client(params)
        else:
            headers = _build_auth_headers(provider.api_key)
            ctx = sse_client(provider.endpoint, headers=headers)

        # Enter the async context manager to get read/write streams
        read, write = await ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        _sessions[key] = session
        _session_contexts[key] = ctx

        logger.debug("Created pooled MCP session for provider %r", provider.name)
        return session


async def _close_all_sessions() -> None:
    """Close all pooled sessions.

    This function must be called from within the background event loop.
    """
    async with _session_lock:
        for key in list(_sessions.keys()):
            try:
                session = _sessions.get(key)
                ctx = _session_contexts.get(key)

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
            except Exception:
                pass  # Best effort cleanup

        count = len(_sessions)
        _sessions.clear()
        _session_contexts.clear()

    if count > 0:
        logger.debug("Closed %d pooled MCP sessions", count)


def _shutdown_mcp_loop() -> None:
    """Shutdown the MCP event loop and close all sessions."""
    global _mcp_loop, _mcp_thread

    if _mcp_loop is None:
        return

    try:
        # Close all sessions
        future = asyncio.run_coroutine_threadsafe(_close_all_sessions(), _mcp_loop)
        try:
            future.result(timeout=5)
        except Exception:
            pass  # Best effort

        # Stop the event loop
        _mcp_loop.call_soon_threadsafe(_mcp_loop.stop)

        # Wait for the thread to finish
        if _mcp_thread is not None:
            _mcp_thread.join(timeout=5)

        logger.debug("Shutdown MCP background event loop")
    except Exception:
        pass  # Best effort cleanup on exit
    finally:
        _mcp_loop = None
        _mcp_thread = None


# Register cleanup on program exit
atexit.register(_shutdown_mcp_loop)


# =============================================================================
# Public API - Session Pool Management
# =============================================================================


def clear_session_pool() -> None:
    """Clear all pooled MCP sessions.

    This closes all active connections and clears the session cache.
    Useful for testing or when provider configurations change.
    """
    global _sessions, _session_contexts

    # Try to close sessions gracefully through the event loop if running
    if _mcp_loop is not None and _mcp_loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(_close_all_sessions(), _mcp_loop)
            future.result(timeout=5)
            return
        except Exception:
            pass  # Fall through to manual cleanup

    # Manual cleanup - just clear the dictionaries
    # This is safe for testing but won't gracefully close connections
    _sessions.clear()
    _session_contexts.clear()


def get_session_pool_info() -> dict[str, Any]:
    """Get information about the session pool.

    Returns:
        Dictionary with pool statistics.
    """
    return {
        "active_sessions": len(_sessions),
        "provider_keys": list(_sessions.keys()),
    }


def clear_provider_caches(providers: list[MCPProviderT]) -> int:
    """Clear all caches for specific MCP providers.

    This function clears both the tools cache and session pool entries for the given
    providers. Use this at job completion to prevent memory buildup in long-running
    services.

    Args:
        providers: List of MCP provider configs to clear caches for.

    Returns:
        Number of cache entries cleared (tools + sessions).
    """
    cleared_count = 0

    for provider in providers:
        key = _provider_cache_key(provider)

        # Clear tools cache for this provider
        if key in _tools_cache:
            del _tools_cache[key]
            cleared_count += 1
        if key in _tools_locks:
            del _tools_locks[key]

        # Clear session for this provider (best effort, async cleanup not guaranteed)
        if key in _sessions:
            del _sessions[key]
            cleared_count += 1
        if key in _session_contexts:
            del _session_contexts[key]

    if cleared_count > 0:
        logger.debug("Cleared %d provider cache entries", cleared_count)

    return cleared_count


# =============================================================================
# Public API - Cached Operations
# =============================================================================


def list_tools(provider: MCPProviderT) -> tuple[MCPToolDefinition, ...]:
    """List tools from an MCP provider (cached with request coalescing).

    Results are cached using the provider's serialized config (including api_key).
    This cache is shared across all MCPFacade instances for efficiency.

    Request coalescing ensures that concurrent requests for the same provider
    share a single in-flight request, preventing thundering herd problems.

    The provider's api_key should already be resolved by the caller - this function
    uses it directly for authentication.

    Args:
        provider: The MCP provider config (with resolved api_key if applicable).

    Returns:
        Tuple of tool definitions from the provider (tuple for hashability).

    Raises:
        MCPToolError: If communication with the provider fails or times out.
    """
    key = _provider_cache_key(provider)

    # Fast path: check cache without lock
    if key in _tools_cache:
        return _tools_cache[key]

    # Slow path: fetch with coalescing
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(
        _list_tools_with_coalescing(provider, key),
        loop,
    )

    try:
        return future.result()
    except TimeoutError as exc:
        raise MCPToolError(f"Timed out while listing tools on {provider.name!r}.") from exc


def clear_tools_cache() -> None:
    """Clear the list_tools cache.

    Useful for testing or when providers need to be refreshed.
    """
    _tools_cache.clear()
    _tools_locks.clear()


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics for list_tools.

    Returns:
        Dictionary with cache statistics.
    """
    return {
        "currsize": len(_tools_cache),
        "providers": list(_tools_cache.keys()),
    }


# =============================================================================
# Public API - Uncached Operations
# =============================================================================


def call_tool(
    provider: MCPProviderT,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout_sec: float | None = None,
) -> MCPToolResult:
    """Call a tool on an MCP provider.

    This operation is NOT cached - each call executes against the MCP server.

    The provider's api_key should already be resolved by the caller - this function
    uses it directly for authentication.

    Args:
        provider: The MCP provider config (with resolved api_key if applicable).
        tool_name: Name of the tool to call.
        arguments: Arguments to pass to the tool.
        timeout_sec: Optional timeout in seconds.

    Returns:
        Result from the tool execution.

    Raises:
        MCPToolError: If the tool call fails or times out.
    """
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(
        _call_tool_async(provider, tool_name, arguments),
        loop,
    )

    try:
        return future.result(timeout=timeout_sec)
    except TimeoutError as exc:
        timeout_label = f"{timeout_sec:.1f}" if timeout_sec is not None else "unknown"
        raise MCPToolError(f"Timed out after {timeout_label}s while calling tool {tool_name!r}.") from exc


def call_tools_parallel(
    calls: list[tuple[MCPProviderT, str, dict[str, Any]]],
    *,
    timeout_sec: float | None = None,
) -> list[MCPToolResult]:
    """Call multiple tools in parallel.

    Executes all tool calls concurrently within the background event loop.
    This is more efficient than sequential calls when multiple tool calls
    are needed (e.g., parallel tool calling from LLMs).

    Args:
        calls: List of (provider, tool_name, arguments) tuples.
        timeout_sec: Optional timeout in seconds for all calls.

    Returns:
        List of results, in the same order as the input calls.

    Raises:
        MCPToolError: If any tool call fails or times out.
    """
    if not calls:
        return []

    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(
        _call_tools_parallel_async(calls),
        loop,
    )

    try:
        return future.result(timeout=timeout_sec)
    except TimeoutError as exc:
        timeout_label = f"{timeout_sec:.1f}" if timeout_sec is not None else "unknown"
        raise MCPToolError(f"Timed out after {timeout_label}s while calling tools in parallel.") from exc


async def _call_tools_parallel_async(
    calls: list[tuple[MCPProviderT, str, dict[str, Any]]],
) -> list[MCPToolResult]:
    """Internal async implementation of parallel tool calling.

    Args:
        calls: List of (provider, tool_name, arguments) tuples.

    Returns:
        List of results in same order as input.
    """
    return await asyncio.gather(*[_call_tool_async(p, n, a) for p, n, a in calls])


# =============================================================================
# Internal - Request Coalescing Implementation
# =============================================================================


async def _list_tools_with_coalescing(
    provider: MCPProviderT,
    key: str,
) -> tuple[MCPToolDefinition, ...]:
    """List tools with request coalescing to prevent thundering herd.

    Uses per-key locks to ensure only one in-flight request per provider.
    Other concurrent callers wait for the first request to complete and
    share the cached result.

    Args:
        provider: The MCP provider config.
        key: Cache key (provider's serialized JSON).

    Returns:
        Tuple of tool definitions.
    """
    from data_designer.engine.mcp.registry import MCPToolDefinition

    # Get or create lock for this key
    async with _tools_locks_lock:
        if key not in _tools_locks:
            _tools_locks[key] = asyncio.Lock()
        lock = _tools_locks[key]

    # Acquire the per-key lock (coalescing happens here)
    async with lock:
        # Double-check cache inside lock - critical for coalescing!
        if key in _tools_cache:
            return _tools_cache[key]

        # Actually fetch (only one caller per key gets here)
        session = await _get_or_create_session(provider)
        result = await session.list_tools()

        raw_tools = getattr(result, "tools", result)
        if not isinstance(raw_tools, list):
            raise MCPToolError("Unexpected response from MCP provider when listing tools.")

        tools = tuple(_coerce_tool_definition(tool, MCPToolDefinition) for tool in raw_tools)

        # Cache the result
        _tools_cache[key] = tools
        logger.debug("Cached tools for provider %r (%d tools)", provider.name, len(tools))

        return tools


# =============================================================================
# Internal - Async Implementations
# =============================================================================


async def _call_tool_async(
    provider: MCPProviderT,
    tool_name: str,
    arguments: dict[str, Any],
) -> MCPToolResult:
    """Call a tool on a provider asynchronously.

    Uses the session pool to reuse connections across calls.

    Args:
        provider: The MCP provider config (with resolved api_key if applicable).
        tool_name: Name of the tool to call.
        arguments: Arguments to pass to the tool.

    Returns:
        Result from the tool execution.
    """
    from data_designer.engine.mcp.registry import MCPToolResult

    session = await _get_or_create_session(provider)
    result = await session.call_tool(tool_name, arguments)

    content = _serialize_tool_result_content(result)
    is_error = getattr(result, "isError", None)
    if is_error is None:
        is_error = getattr(result, "is_error", False)

    return MCPToolResult(content=content, is_error=bool(is_error))


# =============================================================================
# Internal - Helpers
# =============================================================================


def _build_auth_headers(api_key: str | None) -> dict[str, Any] | None:
    """Build authentication headers for SSE client.

    Args:
        api_key: Optional resolved API key.

    Returns:
        Headers dict with Authorization header, or None if no api_key.
    """
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def _coerce_tool_definition(tool: Any, tool_definition_cls: type[MCPToolDefinition]) -> MCPToolDefinition:
    """Coerce a tool from various formats into MCPToolDefinition.

    Args:
        tool: The tool object from the MCP response.
        tool_definition_cls: The MCPToolDefinition class to instantiate.

    Returns:
        A normalized MCPToolDefinition.

    Raises:
        MCPToolError: If the tool has no name.
    """
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
    """Serialize tool result content to a string.

    Handles various result formats including strings, dicts, lists,
    and MCP content objects.

    Args:
        result: The raw result from the MCP tool call.

    Returns:
        The serialized content as a string.
    """
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
