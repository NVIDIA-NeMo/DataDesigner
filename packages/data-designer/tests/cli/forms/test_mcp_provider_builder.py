# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.forms.mcp_provider_builder import MCPProviderFormBuilder
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider

# =============================================================================
# Name validation tests
# =============================================================================


def test_validate_name_rejects_empty_string() -> None:
    """Test name validation rejects empty strings."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_name("")

    assert is_valid is False
    assert "required" in error.lower()


def test_validate_name_rejects_duplicate_name() -> None:
    """Test name validation rejects names that already exist."""
    builder = MCPProviderFormBuilder(existing_names={"server-1", "server-2"})

    is_valid, error = builder._validate_name("server-1")

    assert is_valid is False
    assert "already exists" in error


def test_validate_name_accepts_unique_name() -> None:
    """Test name validation accepts new unique names."""
    builder = MCPProviderFormBuilder(existing_names={"server-1", "server-2"})

    is_valid, error = builder._validate_name("server-3")

    assert is_valid is True
    assert error is None


def test_validate_name_accepts_any_name_when_no_existing() -> None:
    """Test name validation accepts any name when no existing names."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_name("any-name")

    assert is_valid is True
    assert error is None


# =============================================================================
# Endpoint validation tests
# =============================================================================


def test_validate_endpoint_rejects_empty_string() -> None:
    """Test endpoint validation rejects empty strings."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_endpoint("")

    assert is_valid is False
    assert "required" in error.lower()


def test_validate_endpoint_rejects_invalid_url() -> None:
    """Test endpoint validation rejects URLs without http/https protocol."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_endpoint("not-a-url")

    assert is_valid is False
    assert "Invalid URL format" in error


def test_validate_endpoint_accepts_https_url() -> None:
    """Test endpoint validation accepts valid HTTPS URLs."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_endpoint("https://example.com/sse")

    assert is_valid is True
    assert error is None


def test_validate_endpoint_accepts_http_url() -> None:
    """Test endpoint validation accepts valid HTTP URLs."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_endpoint("http://localhost:8080/sse")

    assert is_valid is True
    assert error is None


# =============================================================================
# Command validation tests
# =============================================================================


def test_validate_command_rejects_empty_string() -> None:
    """Test command validation rejects empty strings."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_command("")

    assert is_valid is False
    assert "required" in error.lower()


def test_validate_command_rejects_whitespace_only() -> None:
    """Test command validation rejects whitespace-only strings."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_command("   ")

    assert is_valid is False
    assert "empty" in error.lower()


def test_validate_command_accepts_valid_command() -> None:
    """Test command validation accepts valid commands."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_command("python")

    assert is_valid is True
    assert error is None


def test_validate_command_accepts_command_with_path() -> None:
    """Test command validation accepts commands with paths."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_command("/usr/bin/python3")

    assert is_valid is True
    assert error is None


# =============================================================================
# Provider type selection tests
# =============================================================================


@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows", return_value=None)
def test_select_provider_type_returns_none_on_cancel(mock_select: MagicMock) -> None:
    """Test provider type selection returns None when cancelled."""
    builder = MCPProviderFormBuilder()

    result = builder._select_provider_type()

    assert result is None


@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows", return_value="sse")
def test_select_provider_type_returns_sse(mock_select: MagicMock) -> None:
    """Test provider type selection returns 'sse' for SSE choice."""
    builder = MCPProviderFormBuilder()

    result = builder._select_provider_type()

    assert result == "sse"


@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows", return_value="stdio")
def test_select_provider_type_returns_stdio(mock_select: MagicMock) -> None:
    """Test provider type selection returns 'stdio' for stdio choice."""
    builder = MCPProviderFormBuilder()

    result = builder._select_provider_type()

    assert result == "stdio"


# =============================================================================
# Remote form tests (SSE and Streamable HTTP)
# =============================================================================


def test_run_remote_form_creates_sse_provider() -> None:
    """Test _run_remote_form creates valid MCPProvider with SSE transport."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "my-server",
        "endpoint": "http://localhost:8080/sse",
        "api_key": "secret-key",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_remote_form("sse")

    assert isinstance(result, MCPProvider)
    assert result.name == "my-server"
    assert result.endpoint == "http://localhost:8080/sse"
    assert result.api_key == "secret-key"
    assert result.provider_type == "sse"


def test_run_remote_form_creates_streamable_http_provider() -> None:
    """Test _run_remote_form creates valid MCPProvider with Streamable HTTP transport."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "my-server",
        "endpoint": "https://api.example.com/mcp",
        "api_key": "secret-key",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_remote_form("streamable_http")

    assert isinstance(result, MCPProvider)
    assert result.name == "my-server"
    assert result.endpoint == "https://api.example.com/mcp"
    assert result.api_key == "secret-key"
    assert result.provider_type == "streamable_http"


def test_run_remote_form_returns_none_on_cancel() -> None:
    """Test _run_remote_form returns None when user cancels."""
    builder = MCPProviderFormBuilder()

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = None

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_remote_form("sse")

    assert result is None


def test_run_remote_form_handles_optional_api_key() -> None:
    """Test _run_remote_form handles missing/empty api_key."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "my-server",
        "endpoint": "http://localhost:8080/sse",
        "api_key": "",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_remote_form("sse")

    assert isinstance(result, MCPProvider)
    assert result.api_key is None


def test_run_remote_form_uses_initial_data() -> None:
    """Test _run_remote_form populates form with initial data."""
    builder = MCPProviderFormBuilder()
    initial_data = {
        "name": "existing-server",
        "endpoint": "http://localhost:9090/sse",
        "api_key": "existing-key",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = initial_data

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        builder._run_remote_form("sse", initial_data)

    mock_form.set_values.assert_called_once_with(initial_data)


@patch("data_designer.cli.forms.mcp_provider_builder.print_error")
def test_run_remote_form_handles_exception(mock_print_error: MagicMock) -> None:
    """Test _run_remote_form handles validation exceptions gracefully."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "",  # Invalid: empty name will cause exception
        "endpoint": "invalid",
        "api_key": "",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with (
        patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form),
        patch(
            "data_designer.cli.forms.mcp_provider_builder.MCPProvider",
            side_effect=Exception("Validation error"),
        ),
    ):
        result = builder._run_remote_form("sse")

    assert result is None
    mock_print_error.assert_called()


# =============================================================================
# Stdio form tests
# =============================================================================


def test_run_stdio_form_creates_local_stdio_provider() -> None:
    """Test _run_stdio_form creates valid LocalStdioMCPProvider."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "my-local-server",
        "command": "python",
        "args": "-m,my_server,--port,8080",
        "env": "PYTHONPATH=/app,DEBUG=1",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_stdio_form()

    assert isinstance(result, LocalStdioMCPProvider)
    assert result.name == "my-local-server"
    assert result.command == "python"
    assert result.args == ["-m", "my_server", "--port", "8080"]
    assert result.env == {"PYTHONPATH": "/app", "DEBUG": "1"}


def test_run_stdio_form_returns_none_on_cancel() -> None:
    """Test _run_stdio_form returns None when user cancels."""
    builder = MCPProviderFormBuilder()

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = None

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_stdio_form()

    assert result is None


def test_run_stdio_form_handles_empty_args_and_env() -> None:
    """Test _run_stdio_form handles empty args and env."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "simple-server",
        "command": "node",
        "args": "",
        "env": "",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_stdio_form()

    assert isinstance(result, LocalStdioMCPProvider)
    assert result.args == []
    assert result.env == {}


def test_run_stdio_form_parses_args_correctly() -> None:
    """Test _run_stdio_form parses comma-separated args."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "server",
        "command": "npx",
        "args": "serve , --port , 3000 ",  # With whitespace
        "env": "",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_stdio_form()

    assert result.args == ["serve", "--port", "3000"]


def test_run_stdio_form_parses_env_correctly() -> None:
    """Test _run_stdio_form parses KEY=VALUE env pairs."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "server",
        "command": "python",
        "args": "",
        "env": " KEY1 = value1 , KEY2=value2 ",  # With whitespace
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_stdio_form()

    assert result.env == {"KEY1": "value1", "KEY2": "value2"}


def test_run_stdio_form_handles_env_without_equals() -> None:
    """Test _run_stdio_form skips env entries without = sign."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "server",
        "command": "python",
        "args": "",
        "env": "KEY1=value1,invalid_entry,KEY2=value2",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder._run_stdio_form()

    # Should only include valid KEY=VALUE pairs
    assert result.env == {"KEY1": "value1", "KEY2": "value2"}


def test_run_stdio_form_uses_initial_data() -> None:
    """Test _run_stdio_form populates form with initial data."""
    builder = MCPProviderFormBuilder()
    initial_data = {
        "name": "existing-server",
        "command": "python",
        "args": ["-m", "server"],
        "env": {"DEBUG": "1"},
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = {
        "name": "existing-server",
        "command": "python",
        "args": "-m,server",
        "env": "DEBUG=1",
    }

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        builder._run_stdio_form(initial_data)

    # Check that set_values was called with converted values
    mock_form.set_values.assert_called_once()
    call_args = mock_form.set_values.call_args[0][0]
    assert call_args["args"] == "-m,server"
    assert call_args["env"] == "DEBUG=1"


@patch("data_designer.cli.forms.mcp_provider_builder.print_error")
def test_run_stdio_form_handles_exception(mock_print_error: MagicMock) -> None:
    """Test _run_stdio_form handles validation exceptions gracefully."""
    builder = MCPProviderFormBuilder()
    form_result = {
        "name": "server",
        "command": "python",
        "args": "",
        "env": "",
    }

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with (
        patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form),
        patch(
            "data_designer.cli.forms.mcp_provider_builder.LocalStdioMCPProvider",
            side_effect=Exception("Validation error"),
        ),
    ):
        result = builder._run_stdio_form()

    assert result is None
    mock_print_error.assert_called()


# =============================================================================
# Full run() workflow tests
# =============================================================================


@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows")
@patch("data_designer.cli.forms.mcp_provider_builder.print_header")
def test_run_with_sse_provider(mock_print_header: MagicMock, mock_select: MagicMock) -> None:
    """Test run() workflow for SSE provider."""
    builder = MCPProviderFormBuilder()
    mock_select.return_value = "sse"

    form_result = {
        "name": "sse-server",
        "endpoint": "http://localhost:8080/sse",
        "api_key": "key",
    }
    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder.run()

    assert isinstance(result, MCPProvider)
    assert result.name == "sse-server"


@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows")
@patch("data_designer.cli.forms.mcp_provider_builder.print_header")
def test_run_with_stdio_provider(mock_print_header: MagicMock, mock_select: MagicMock) -> None:
    """Test run() workflow for stdio provider."""
    builder = MCPProviderFormBuilder()
    mock_select.return_value = "stdio"

    form_result = {
        "name": "stdio-server",
        "command": "python",
        "args": "",
        "env": "",
    }
    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder.run()

    assert isinstance(result, LocalStdioMCPProvider)
    assert result.name == "stdio-server"


@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows", return_value=None)
@patch("data_designer.cli.forms.mcp_provider_builder.print_header")
def test_run_returns_none_on_type_cancel(mock_print_header: MagicMock, mock_select: MagicMock) -> None:
    """Test run() returns None when provider type selection is cancelled."""
    builder = MCPProviderFormBuilder()

    result = builder.run()

    assert result is None


@patch("data_designer.cli.forms.mcp_provider_builder.confirm_action", return_value=False)
@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows")
@patch("data_designer.cli.forms.mcp_provider_builder.print_header")
def test_run_returns_none_on_form_cancel_and_no_retry(
    mock_print_header: MagicMock,
    mock_select: MagicMock,
    mock_confirm: MagicMock,
) -> None:
    """Test run() returns None when form is cancelled and user declines retry."""
    builder = MCPProviderFormBuilder()
    mock_select.return_value = "sse"

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = None

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder.run()

    assert result is None
    mock_confirm.assert_called_once_with("Try a different provider type?", default=False)


@patch("data_designer.cli.forms.mcp_provider_builder.select_with_arrows")
@patch("data_designer.cli.forms.mcp_provider_builder.print_header")
def test_run_uses_provider_type_from_initial_data(mock_print_header: MagicMock, mock_select: MagicMock) -> None:
    """Test run() uses provider_type from initial_data without prompting."""
    builder = MCPProviderFormBuilder()
    initial_data = {
        "provider_type": "sse",
        "name": "existing",
        "endpoint": "http://localhost:8080/sse",
    }

    form_result = {
        "name": "existing",
        "endpoint": "http://localhost:8080/sse",
        "api_key": "",
    }
    mock_form = MagicMock()
    mock_form.prompt_all.return_value = form_result

    with patch("data_designer.cli.forms.mcp_provider_builder.Form", return_value=mock_form):
        result = builder.run(initial_data)

    # select_with_arrows should not be called for provider type
    mock_select.assert_not_called()
    assert isinstance(result, MCPProvider)


# =============================================================================
# Edge cases and integration
# =============================================================================


def test_builder_title_is_set() -> None:
    """Test builder has appropriate title."""
    builder = MCPProviderFormBuilder()
    assert builder.title == "MCP Provider Configuration"


def test_existing_names_defaults_to_empty_set() -> None:
    """Test existing_names defaults to empty set when not provided."""
    builder = MCPProviderFormBuilder()
    assert builder.existing_names == set()


def test_existing_names_is_stored() -> None:
    """Test existing_names is stored when provided."""
    names = {"server-1", "server-2"}
    builder = MCPProviderFormBuilder(existing_names=names)
    assert builder.existing_names == names


def test_validate_endpoint_rejects_url_without_protocol() -> None:
    """Test endpoint validation catches URLs missing protocol."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_endpoint("example.com/sse")

    assert is_valid is False
    assert "http" in error.lower()


def test_validate_endpoint_accepts_localhost() -> None:
    """Test endpoint validation accepts localhost URLs."""
    builder = MCPProviderFormBuilder()

    is_valid, error = builder._validate_endpoint("http://localhost:8080/sse")

    assert is_valid is True
    assert error is None
