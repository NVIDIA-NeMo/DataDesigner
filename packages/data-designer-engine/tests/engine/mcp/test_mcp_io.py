# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Iterator

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider
from data_designer.engine.mcp import io as mcp_io
from data_designer.engine.mcp.errors import MCPToolError
from data_designer.engine.mcp.registry import MCPToolDefinition, MCPToolResult


@pytest.fixture(autouse=True)
def clear_session_pool_before_tests() -> Iterator[None]:
    """Clear the session pool before each test for isolation."""
    mcp_io.clear_session_pool()
    yield
    mcp_io.clear_session_pool()


@pytest.fixture
def stub_stdio_provider() -> LocalStdioMCPProvider:
    """Create a stub stdio MCP provider for testing."""
    return LocalStdioMCPProvider(
        name="test-stdio",
        command="python",
        args=["-m", "test_server"],
        env={"TEST_VAR": "value"},
    )


@pytest.fixture
def stub_sse_provider() -> MCPProvider:
    """Create a stub SSE MCP provider for testing."""
    return MCPProvider(
        name="test-sse",
        endpoint="http://localhost:8080/sse",
        api_key="test-key",
    )


# =============================================================================
# Cache operations tests
# =============================================================================


def test_clear_tools_cache() -> None:
    """Test that clear_tools_cache clears the cache."""
    # Just verify it doesn't raise
    mcp_io.clear_tools_cache()


def test_get_cache_info() -> None:
    """Test that get_cache_info returns cache statistics."""
    mcp_io.clear_tools_cache()  # Ensure clean state
    info = mcp_io.get_cache_info()

    assert "currsize" in info
    assert "providers" in info
    assert info["currsize"] == 0
    assert info["providers"] == []


# =============================================================================
# Tool definition coercion tests
# =============================================================================


def test_coerce_tool_definition_from_dict() -> None:
    """Test coercing a tool definition from a dictionary."""
    tool = {
        "name": "lookup",
        "description": "Lookup tool",
        "inputSchema": {"type": "object", "properties": {}},
    }

    result = mcp_io._coerce_tool_definition(tool, MCPToolDefinition)

    assert result.name == "lookup"
    assert result.description == "Lookup tool"
    assert result.input_schema == {"type": "object", "properties": {}}


def test_coerce_tool_definition_from_dict_snake_case() -> None:
    """Test coercing a tool definition with snake_case input_schema."""
    tool = {
        "name": "search",
        "description": "Search tool",
        "input_schema": {"type": "object"},
    }

    result = mcp_io._coerce_tool_definition(tool, MCPToolDefinition)

    assert result.name == "search"
    assert result.input_schema == {"type": "object"}


def test_coerce_tool_definition_from_object() -> None:
    """Test coercing a tool definition from an object."""

    class FakeTool:
        name = "fetch"
        description = "Fetch tool"
        inputSchema = {"type": "object"}

    result = mcp_io._coerce_tool_definition(FakeTool(), MCPToolDefinition)

    assert result.name == "fetch"
    assert result.description == "Fetch tool"
    assert result.input_schema == {"type": "object"}


def test_coerce_tool_definition_missing_name() -> None:
    """Test that missing name raises MCPToolError."""
    tool = {"description": "No name tool"}

    with pytest.raises(MCPToolError, match="without a name"):
        mcp_io._coerce_tool_definition(tool, MCPToolDefinition)


# =============================================================================
# Tool result serialization tests
# =============================================================================


def test_serialize_content_none() -> None:
    """Test serializing None content."""

    class FakeResult:
        content = None

    assert mcp_io._serialize_tool_result_content(FakeResult()) == ""


def test_serialize_content_string() -> None:
    """Test serializing string content."""

    class FakeResult:
        content = "Hello, world!"

    assert mcp_io._serialize_tool_result_content(FakeResult()) == "Hello, world!"


def test_serialize_content_dict() -> None:
    """Test serializing dict content."""

    class FakeResult:
        content = {"key": "value"}

    assert mcp_io._serialize_tool_result_content(FakeResult()) == '{"key": "value"}'


def test_serialize_content_list_of_strings() -> None:
    """Test serializing list of strings content."""

    class FakeResult:
        content = ["line1", "line2", "line3"]

    assert mcp_io._serialize_tool_result_content(FakeResult()) == "line1\nline2\nline3"


def test_serialize_content_list_of_text_items() -> None:
    """Test serializing list of text items."""

    class FakeResult:
        content = [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}]

    assert mcp_io._serialize_tool_result_content(FakeResult()) == "First\nSecond"


def test_serialize_content_list_of_dicts() -> None:
    """Test serializing list of non-text dicts."""

    class FakeResult:
        content = [{"type": "data", "value": 1}]

    result = mcp_io._serialize_tool_result_content(FakeResult())
    assert '{"type": "data", "value": 1}' in result


def test_serialize_content_list_with_objects() -> None:
    """Test serializing list with objects that have text attribute."""

    class TextItem:
        text = "Object text"

    class FakeResult:
        content = [TextItem()]

    assert mcp_io._serialize_tool_result_content(FakeResult()) == "Object text"


def test_serialize_content_fallback_to_str() -> None:
    """Test serializing content falls back to str()."""

    class FakeResult:
        content = 12345

    assert mcp_io._serialize_tool_result_content(FakeResult()) == "12345"


# =============================================================================
# list_tools caching tests (mocked)
# =============================================================================


def test_list_tools_uses_cache(monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider) -> None:
    """Test that list_tools uses caching with request coalescing."""
    import asyncio

    call_count = 0

    async def mock_list_tools_with_coalescing(provider: Any, key: str) -> tuple[MCPToolDefinition, ...]:
        nonlocal call_count
        call_count += 1
        return (MCPToolDefinition(name="tool1", description="Tool 1", input_schema={}),)

    def mock_ensure_loop() -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    class MockFuture:
        def __init__(self, result: Any) -> None:
            self._result = result

        def result(self, timeout: float | None = None) -> Any:
            return self._result

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        if hasattr(coro, "close"):
            coro.close()
        return MockFuture((MCPToolDefinition(name="tool1", description="Tool 1", input_schema={}),))

    # Clear cache to ensure we hit the slow path
    mcp_io.clear_tools_cache()

    monkeypatch.setattr(mcp_io, "_ensure_loop", mock_ensure_loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    result = mcp_io.list_tools(stub_stdio_provider)

    assert len(result) == 1
    assert result[0].name == "tool1"


# =============================================================================
# call_tool tests (mocked)
# =============================================================================


def test_call_tool_uses_background_loop(
    monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider
) -> None:
    """Test that call_tool uses the background event loop."""
    import asyncio

    call_tool_async_called = False
    ensure_loop_called = False

    async def mock_call_tool_async(provider: Any, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        nonlocal call_tool_async_called
        call_tool_async_called = True
        return MCPToolResult(content="mocked result", is_error=False)

    def mock_ensure_loop() -> asyncio.AbstractEventLoop:
        nonlocal ensure_loop_called
        ensure_loop_called = True
        # Create a real event loop for the test
        loop = asyncio.new_event_loop()
        return loop

    # Mock run_coroutine_threadsafe to return a mock future
    class MockFuture:
        def __init__(self, result: Any) -> None:
            self._result = result

        def result(self, timeout: float | None = None) -> Any:
            return self._result

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        # Close the coroutine to avoid warning
        coro.close()
        return MockFuture(MCPToolResult(content="mocked result", is_error=False))

    monkeypatch.setattr(mcp_io, "_ensure_loop", mock_ensure_loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    result = mcp_io.call_tool(stub_stdio_provider, "test_tool", {"arg": "value"}, timeout_sec=30.0)

    assert ensure_loop_called
    assert result.content == "mocked result"
    assert result.is_error is False


# =============================================================================
# call_tools_parallel tests
# =============================================================================


def test_call_tools_parallel_empty_list() -> None:
    """Test that call_tools_parallel returns empty list for empty input."""
    result = mcp_io.call_tools_parallel([])
    assert result == []


def test_call_tools_parallel_uses_background_loop(
    monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider
) -> None:
    """Test that call_tools_parallel uses the background event loop."""
    import asyncio

    ensure_loop_called = False

    def mock_ensure_loop() -> asyncio.AbstractEventLoop:
        nonlocal ensure_loop_called
        ensure_loop_called = True
        loop = asyncio.new_event_loop()
        return loop

    class MockFuture:
        def __init__(self, results: list[MCPToolResult]) -> None:
            self._results = results

        def result(self, timeout: float | None = None) -> list[MCPToolResult]:
            return self._results

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        # coro could be a gather future or coroutine, handle both
        if hasattr(coro, "close"):
            coro.close()
        elif hasattr(coro, "cancel"):
            coro.cancel()
        return MockFuture(
            [
                MCPToolResult(content="result1", is_error=False),
                MCPToolResult(content="result2", is_error=False),
            ]
        )

    monkeypatch.setattr(mcp_io, "_ensure_loop", mock_ensure_loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    calls = [
        (stub_stdio_provider, "tool1", {"arg": "value1"}),
        (stub_stdio_provider, "tool2", {"arg": "value2"}),
    ]
    results = mcp_io.call_tools_parallel(calls, timeout_sec=30.0)

    assert ensure_loop_called
    assert len(results) == 2
    assert results[0].content == "result1"
    assert results[1].content == "result2"


def test_call_tools_parallel_timeout(
    monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider
) -> None:
    """Test that call_tools_parallel raises MCPToolError on timeout."""
    import asyncio

    def mock_ensure_loop() -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    class MockFuture:
        def result(self, timeout: float | None = None) -> None:
            raise TimeoutError()

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        # coro could be a gather future or coroutine, handle both
        if hasattr(coro, "close"):
            coro.close()
        elif hasattr(coro, "cancel"):
            coro.cancel()
        return MockFuture()

    monkeypatch.setattr(mcp_io, "_ensure_loop", mock_ensure_loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    calls = [(stub_stdio_provider, "tool1", {})]

    with pytest.raises(MCPToolError, match="Timed out.*while calling tools in parallel"):
        mcp_io.call_tools_parallel(calls, timeout_sec=1.0)


# =============================================================================
# Session pool tests
# =============================================================================


def test_clear_session_pool() -> None:
    """Test that clear_session_pool clears the pool."""
    # Just verify it doesn't raise
    mcp_io.clear_session_pool()


def test_get_session_pool_info() -> None:
    """Test that get_session_pool_info returns pool statistics."""
    # Clear pool first to get a clean state
    mcp_io.clear_session_pool()

    info = mcp_io.get_session_pool_info()

    assert "active_sessions" in info
    assert "provider_keys" in info
    assert info["active_sessions"] == 0
    assert info["provider_keys"] == []


@pytest.mark.asyncio
async def test_get_or_create_session_reuses_session(
    monkeypatch: pytest.MonkeyPatch, stub_sse_provider: MCPProvider
) -> None:
    """Test that _get_or_create_session reuses existing sessions."""
    # Clear pool first
    mcp_io.clear_session_pool()

    # Track how many times we enter sse_client
    enter_count = 0

    class MockContextManager:
        async def __aenter__(self) -> tuple[Any, Any]:
            nonlocal enter_count
            enter_count += 1
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            pass

    class MockSession:
        async def __aenter__(self) -> "MockSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            pass

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManager:
        return MockContextManager()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSession())

    # First call should create a new session
    session1 = await mcp_io._get_or_create_session(stub_sse_provider)
    assert enter_count == 1

    # Second call should reuse the session
    session2 = await mcp_io._get_or_create_session(stub_sse_provider)
    assert enter_count == 1  # Still 1, didn't create a new one

    assert session1 is session2

    # Clean up
    mcp_io.clear_session_pool()


@pytest.mark.asyncio
async def test_get_or_create_session_different_providers_get_different_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that different providers get different sessions."""
    # Clear pool first
    mcp_io.clear_session_pool()

    created_sessions: list[Any] = []

    class MockContextManager:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            pass

    class MockSession:
        async def __aenter__(self) -> "MockSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            pass

    def create_mock_session(r: Any, w: Any) -> MockSession:
        session = MockSession()
        created_sessions.append(session)
        return session

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManager:
        return MockContextManager()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", create_mock_session)

    provider1 = MCPProvider(name="provider1", endpoint="http://localhost:8080/sse", api_key="key1")
    provider2 = MCPProvider(name="provider2", endpoint="http://localhost:8081/sse", api_key="key2")

    session1 = await mcp_io._get_or_create_session(provider1)
    session2 = await mcp_io._get_or_create_session(provider2)

    # Different providers should get different sessions
    assert session1 is not session2
    assert len(created_sessions) == 2

    # Verify pool info
    info = mcp_io.get_session_pool_info()
    assert info["active_sessions"] == 2

    # Clean up
    mcp_io.clear_session_pool()


def test_session_pool_info_after_creating_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_session_pool_info reflects created sessions."""
    import asyncio

    # Clear pool first
    mcp_io.clear_session_pool()

    class MockContextManager:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            pass

    class MockSession:
        async def __aenter__(self) -> "MockSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            pass

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManager:
        return MockContextManager()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSession())

    provider = MCPProvider(name="test", endpoint="http://localhost:8080/sse", api_key="key")

    async def create_session() -> None:
        await mcp_io._get_or_create_session(provider)

    asyncio.run(create_session())

    info = mcp_io.get_session_pool_info()
    assert info["active_sessions"] == 1
    assert len(info["provider_keys"]) == 1
    assert provider.model_dump_json() in info["provider_keys"]

    # Clean up
    mcp_io.clear_session_pool()

    # Verify cleared
    info = mcp_io.get_session_pool_info()
    assert info["active_sessions"] == 0


# =============================================================================
# Background event loop tests
# =============================================================================


def test_ensure_loop_creates_background_loop() -> None:
    """Test that _ensure_loop creates a background event loop."""
    # This tests the real _ensure_loop
    loop = mcp_io._ensure_loop()

    assert loop is not None
    assert loop.is_running()


def test_ensure_loop_returns_same_loop() -> None:
    """Test that _ensure_loop returns the same loop on subsequent calls."""
    loop1 = mcp_io._ensure_loop()
    loop2 = mcp_io._ensure_loop()

    assert loop1 is loop2


# =============================================================================
# Request coalescing tests
# =============================================================================


@pytest.mark.asyncio
async def test_list_tools_coalescing_reuses_cached_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that request coalescing returns cached results for same provider."""
    mcp_io.clear_tools_cache()
    mcp_io.clear_session_pool()

    fetch_count = 0

    class MockContextManager:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            pass

    class MockSession:
        async def __aenter__(self) -> "MockSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def list_tools(self) -> Any:
            nonlocal fetch_count
            fetch_count += 1

            class FakeResult:
                tools = [{"name": "tool1", "description": "Tool 1", "inputSchema": {}}]

            return FakeResult()

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManager:
        return MockContextManager()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSession())

    provider = MCPProvider(name="test", endpoint="http://localhost:8080/sse", api_key="key")
    key = provider.model_dump_json()

    # First call should fetch
    result1 = await mcp_io._list_tools_with_coalescing(provider, key)
    assert fetch_count == 1
    assert len(result1) == 1
    assert result1[0].name == "tool1"

    # Second call should use cache (no additional fetch)
    result2 = await mcp_io._list_tools_with_coalescing(provider, key)
    assert fetch_count == 1  # Still 1
    assert result1 == result2

    # Verify cache info
    info = mcp_io.get_cache_info()
    assert info["currsize"] == 1

    mcp_io.clear_tools_cache()
    mcp_io.clear_session_pool()


@pytest.mark.asyncio
async def test_list_tools_coalescing_different_providers_fetch_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that different providers fetch independently."""
    mcp_io.clear_tools_cache()
    mcp_io.clear_session_pool()

    fetch_count = 0

    class MockContextManager:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            pass

    class MockSession:
        async def __aenter__(self) -> "MockSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def list_tools(self) -> Any:
            nonlocal fetch_count
            fetch_count += 1

            class FakeResult:
                tools = [{"name": f"tool{fetch_count}", "description": f"Tool {fetch_count}", "inputSchema": {}}]

            return FakeResult()

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManager:
        return MockContextManager()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSession())

    provider1 = MCPProvider(name="provider1", endpoint="http://localhost:8080/sse", api_key="key1")
    provider2 = MCPProvider(name="provider2", endpoint="http://localhost:8081/sse", api_key="key2")

    # Each provider should fetch independently
    result1 = await mcp_io._list_tools_with_coalescing(provider1, provider1.model_dump_json())
    assert fetch_count == 1

    result2 = await mcp_io._list_tools_with_coalescing(provider2, provider2.model_dump_json())
    assert fetch_count == 2

    # Different providers got different results
    assert result1[0].name == "tool1"
    assert result2[0].name == "tool2"

    # Verify cache has both providers
    info = mcp_io.get_cache_info()
    assert info["currsize"] == 2

    mcp_io.clear_tools_cache()
    mcp_io.clear_session_pool()


# =============================================================================
# Stdio session creation tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_or_create_session_for_stdio_provider(
    monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider
) -> None:
    """Test that _get_or_create_session creates session for stdio provider."""
    mcp_io.clear_session_pool()

    class MockContextManager:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            pass

    class MockSession:
        async def __aenter__(self) -> "MockSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            pass

    stdio_client_called = False
    received_params: Any = None

    def mock_stdio_client(params: Any) -> MockContextManager:
        nonlocal stdio_client_called, received_params
        stdio_client_called = True
        received_params = params
        return MockContextManager()

    monkeypatch.setattr(mcp_io, "stdio_client", mock_stdio_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSession())

    session = await mcp_io._get_or_create_session(stub_stdio_provider)

    assert stdio_client_called
    assert received_params.command == stub_stdio_provider.command
    assert received_params.args == stub_stdio_provider.args
    assert received_params.env == stub_stdio_provider.env
    assert session is not None

    mcp_io.clear_session_pool()


# =============================================================================
# Session cleanup exception handling tests
# =============================================================================


@pytest.mark.asyncio
async def test_close_all_sessions_handles_session_aexit_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that _close_all_sessions handles exceptions during session.__aexit__."""
    mcp_io.clear_session_pool()

    class MockContextManager:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            pass

    class MockSessionWithException:
        async def __aenter__(self) -> "MockSessionWithException":
            return self

        async def __aexit__(self, *args: Any) -> None:
            raise RuntimeError("Session cleanup error")

        async def initialize(self) -> None:
            pass

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManager:
        return MockContextManager()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSessionWithException())

    provider = MCPProvider(name="test", endpoint="http://localhost:8080/sse", api_key="key")
    await mcp_io._get_or_create_session(provider)

    # Verify session was created
    assert mcp_io.get_session_pool_info()["active_sessions"] == 1

    # Close should handle exception gracefully
    await mcp_io._close_all_sessions()

    # Pool should be cleared
    assert mcp_io.get_session_pool_info()["active_sessions"] == 0


@pytest.mark.asyncio
async def test_close_all_sessions_handles_context_aexit_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that _close_all_sessions handles exceptions during ctx.__aexit__."""
    mcp_io.clear_session_pool()

    class MockContextManagerWithException:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            raise RuntimeError("Context cleanup error")

    class MockSession:
        async def __aenter__(self) -> "MockSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            pass

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManagerWithException:
        return MockContextManagerWithException()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSession())

    provider = MCPProvider(name="test", endpoint="http://localhost:8080/sse", api_key="key")
    await mcp_io._get_or_create_session(provider)

    # Verify session was created
    assert mcp_io.get_session_pool_info()["active_sessions"] == 1

    # Close should handle exception gracefully
    await mcp_io._close_all_sessions()

    # Pool should be cleared
    assert mcp_io.get_session_pool_info()["active_sessions"] == 0


@pytest.mark.asyncio
async def test_close_all_sessions_handles_both_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that _close_all_sessions handles exceptions from both session and context."""
    mcp_io.clear_session_pool()

    class MockContextManagerWithException:
        async def __aenter__(self) -> tuple[Any, Any]:
            return ("mock_read", "mock_write")

        async def __aexit__(self, *args: Any) -> None:
            raise RuntimeError("Context cleanup error")

    class MockSessionWithException:
        async def __aenter__(self) -> "MockSessionWithException":
            return self

        async def __aexit__(self, *args: Any) -> None:
            raise RuntimeError("Session cleanup error")

        async def initialize(self) -> None:
            pass

    def mock_sse_client(endpoint: str, headers: dict[str, Any] | None = None) -> MockContextManagerWithException:
        return MockContextManagerWithException()

    monkeypatch.setattr(mcp_io, "sse_client", mock_sse_client)
    monkeypatch.setattr(mcp_io, "ClientSession", lambda r, w: MockSessionWithException())

    provider = MCPProvider(name="test", endpoint="http://localhost:8080/sse", api_key="key")
    await mcp_io._get_or_create_session(provider)

    # Close should handle both exceptions gracefully
    await mcp_io._close_all_sessions()

    # Pool should be cleared
    assert mcp_io.get_session_pool_info()["active_sessions"] == 0


@pytest.mark.asyncio
async def test_close_all_sessions_handles_outer_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that _close_all_sessions handles outer exception during iteration."""
    mcp_io.clear_session_pool()

    # Manually inject sessions that will cause issues
    mcp_io._sessions["bad_key"] = None
    mcp_io._session_contexts["bad_key"] = None

    # Should not raise
    await mcp_io._close_all_sessions()

    assert mcp_io.get_session_pool_info()["active_sessions"] == 0


# =============================================================================
# Shutdown loop tests
# =============================================================================


def test_shutdown_mcp_loop_when_loop_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _shutdown_mcp_loop returns early when _mcp_loop is None."""
    # Temporarily set _mcp_loop to None
    original_loop = mcp_io._mcp_loop
    monkeypatch.setattr(mcp_io, "_mcp_loop", None)

    # Should not raise
    mcp_io._shutdown_mcp_loop()

    # Restore
    monkeypatch.setattr(mcp_io, "_mcp_loop", original_loop)


def test_shutdown_mcp_loop_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _shutdown_mcp_loop handles exceptions gracefully."""
    import asyncio
    import threading
    from unittest.mock import MagicMock

    # Create a mock loop that raises on operations
    mock_loop = MagicMock()
    mock_loop.is_running.return_value = True

    mock_thread = MagicMock(spec=threading.Thread)

    def raise_exception(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Shutdown error")

    mock_loop.call_soon_threadsafe = raise_exception

    # Mock run_coroutine_threadsafe to raise
    class MockFuture:
        def result(self, timeout: float | None = None) -> None:
            raise RuntimeError("Future error")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coro, loop: MockFuture())

    original_loop = mcp_io._mcp_loop
    original_thread = mcp_io._mcp_thread

    monkeypatch.setattr(mcp_io, "_mcp_loop", mock_loop)
    monkeypatch.setattr(mcp_io, "_mcp_thread", mock_thread)

    # Should not raise despite exceptions
    mcp_io._shutdown_mcp_loop()

    # Should reset to None in finally block
    assert mcp_io._mcp_loop is None
    assert mcp_io._mcp_thread is None

    # Restore for other tests
    mcp_io._mcp_loop = original_loop
    mcp_io._mcp_thread = original_thread


def test_shutdown_mcp_loop_joins_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _shutdown_mcp_loop joins the thread when present."""
    import asyncio
    import threading
    from unittest.mock import MagicMock

    mock_loop = MagicMock()
    mock_loop.is_running.return_value = True

    mock_thread = MagicMock(spec=threading.Thread)

    class MockFuture:
        def result(self, timeout: float | None = None) -> None:
            pass

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        if hasattr(coro, "close"):
            coro.close()
        return MockFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    original_loop = mcp_io._mcp_loop
    original_thread = mcp_io._mcp_thread

    monkeypatch.setattr(mcp_io, "_mcp_loop", mock_loop)
    monkeypatch.setattr(mcp_io, "_mcp_thread", mock_thread)

    mcp_io._shutdown_mcp_loop()

    # Verify thread.join was called
    mock_thread.join.assert_called_once_with(timeout=5)

    # Restore for other tests
    mcp_io._mcp_loop = original_loop
    mcp_io._mcp_thread = original_thread


def test_clear_session_pool_without_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clear_session_pool when loop is not running."""
    from unittest.mock import MagicMock

    # Add some entries to the pool manually
    mcp_io._sessions["test_key"] = MagicMock()
    mcp_io._session_contexts["test_key"] = MagicMock()

    # Mock _mcp_loop to be None (not running)
    original_loop = mcp_io._mcp_loop
    monkeypatch.setattr(mcp_io, "_mcp_loop", None)

    # Should clear manually without async cleanup
    mcp_io.clear_session_pool()

    assert len(mcp_io._sessions) == 0
    assert len(mcp_io._session_contexts) == 0

    # Restore
    monkeypatch.setattr(mcp_io, "_mcp_loop", original_loop)


def test_clear_session_pool_with_failing_async_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clear_session_pool falls back to manual cleanup on async failure."""
    import asyncio
    from unittest.mock import MagicMock

    # Add some entries to the pool manually
    mcp_io._sessions["test_key"] = MagicMock()
    mcp_io._session_contexts["test_key"] = MagicMock()

    mock_loop = MagicMock()
    mock_loop.is_running.return_value = True

    class MockFuture:
        def result(self, timeout: float | None = None) -> None:
            raise RuntimeError("Async cleanup failed")

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        if hasattr(coro, "close"):
            coro.close()
        return MockFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    original_loop = mcp_io._mcp_loop
    monkeypatch.setattr(mcp_io, "_mcp_loop", mock_loop)

    # Should fall back to manual cleanup
    mcp_io.clear_session_pool()

    assert len(mcp_io._sessions) == 0
    assert len(mcp_io._session_contexts) == 0

    # Restore
    monkeypatch.setattr(mcp_io, "_mcp_loop", original_loop)


# =============================================================================
# Additional helper function tests
# =============================================================================


def test_build_auth_headers_with_none() -> None:
    """Test _build_auth_headers returns None when api_key is None."""
    result = mcp_io._build_auth_headers(None)
    assert result is None


def test_build_auth_headers_with_empty_string() -> None:
    """Test _build_auth_headers returns None when api_key is empty string."""
    result = mcp_io._build_auth_headers("")
    assert result is None


def test_build_auth_headers_with_key() -> None:
    """Test _build_auth_headers returns proper headers when api_key is provided."""
    result = mcp_io._build_auth_headers("my-secret-key")
    assert result == {"Authorization": "Bearer my-secret-key"}


def test_list_tools_timeout_error(monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider) -> None:
    """Test list_tools raises MCPToolError on timeout."""
    import asyncio

    mcp_io.clear_tools_cache()

    def mock_ensure_loop() -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    class MockFuture:
        def result(self, timeout: float | None = None) -> None:
            raise TimeoutError()

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        if hasattr(coro, "close"):
            coro.close()
        return MockFuture()

    monkeypatch.setattr(mcp_io, "_ensure_loop", mock_ensure_loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    with pytest.raises(MCPToolError, match="Timed out"):
        mcp_io.list_tools(stub_stdio_provider)


def test_call_tool_timeout_error(monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider) -> None:
    """Test call_tool raises MCPToolError on timeout with proper message."""
    import asyncio

    def mock_ensure_loop() -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    class MockFuture:
        def result(self, timeout: float | None = None) -> None:
            raise TimeoutError()

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        if hasattr(coro, "close"):
            coro.close()
        return MockFuture()

    monkeypatch.setattr(mcp_io, "_ensure_loop", mock_ensure_loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    with pytest.raises(MCPToolError, match="Timed out after 10.0s"):
        mcp_io.call_tool(stub_stdio_provider, "test_tool", {}, timeout_sec=10.0)


def test_call_tool_timeout_error_without_timeout_value(
    monkeypatch: pytest.MonkeyPatch, stub_stdio_provider: LocalStdioMCPProvider
) -> None:
    """Test call_tool timeout message when timeout_sec is None."""
    import asyncio

    def mock_ensure_loop() -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    class MockFuture:
        def result(self, timeout: float | None = None) -> None:
            raise TimeoutError()

    def mock_run_coroutine_threadsafe(coro: Any, loop: Any) -> MockFuture:
        if hasattr(coro, "close"):
            coro.close()
        return MockFuture()

    monkeypatch.setattr(mcp_io, "_ensure_loop", mock_ensure_loop)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", mock_run_coroutine_threadsafe)

    with pytest.raises(MCPToolError, match="Timed out after unknown"):
        mcp_io.call_tool(stub_stdio_provider, "test_tool", {}, timeout_sec=None)
