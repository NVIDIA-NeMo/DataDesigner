# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.forms.tool_builder import ToolFormBuilder
from data_designer.config.mcp import ToolConfig

# =============================================================================
# Alias validation tests
# =============================================================================


def test_validate_alias_rejects_empty_string() -> None:
    """Test alias validation rejects empty strings."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_alias("")

    assert is_valid is False
    assert "required" in error.lower()


def test_validate_alias_rejects_duplicate_alias() -> None:
    """Test alias validation rejects aliases that already exist."""
    builder = ToolFormBuilder(existing_aliases={"tool-1", "tool-2"})

    is_valid, error = builder._validate_alias("tool-1")

    assert is_valid is False
    assert "already exists" in error


def test_validate_alias_accepts_unique_alias() -> None:
    """Test alias validation accepts new unique aliases."""
    builder = ToolFormBuilder(existing_aliases={"tool-1", "tool-2"})

    is_valid, error = builder._validate_alias("tool-3")

    assert is_valid is True
    assert error is None


def test_validate_alias_accepts_any_alias_when_no_existing() -> None:
    """Test alias validation accepts any alias when no existing aliases."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_alias("any-alias")

    assert is_valid is True
    assert error is None


# =============================================================================
# Max tool call turns validation tests
# =============================================================================


def test_validate_max_tool_call_turns_accepts_empty() -> None:
    """Test max_tool_call_turns validation accepts empty string (uses default)."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_max_tool_call_turns("")

    assert is_valid is True
    assert error is None


def test_validate_max_tool_call_turns_accepts_positive_integer() -> None:
    """Test max_tool_call_turns validation accepts positive integers."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_max_tool_call_turns("5")

    assert is_valid is True
    assert error is None


def test_validate_max_tool_call_turns_rejects_zero() -> None:
    """Test max_tool_call_turns validation rejects zero."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_max_tool_call_turns("0")

    assert is_valid is False
    assert "at least 1" in error.lower()


def test_validate_max_tool_call_turns_rejects_negative() -> None:
    """Test max_tool_call_turns validation rejects negative values."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_max_tool_call_turns("-1")

    assert is_valid is False
    assert "at least 1" in error.lower()


def test_validate_max_tool_call_turns_rejects_non_integer() -> None:
    """Test max_tool_call_turns validation rejects non-integer values."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_max_tool_call_turns("abc")

    assert is_valid is False
    assert "positive integer" in error.lower()


def test_validate_max_tool_call_turns_rejects_float() -> None:
    """Test max_tool_call_turns validation rejects float values."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_max_tool_call_turns("5.5")

    assert is_valid is False
    assert "positive integer" in error.lower()


# =============================================================================
# Timeout validation tests
# =============================================================================


def test_validate_timeout_accepts_empty() -> None:
    """Test timeout validation accepts empty string (no timeout)."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_timeout("")

    assert is_valid is True
    assert error is None


def test_validate_timeout_accepts_positive_number() -> None:
    """Test timeout validation accepts positive numbers."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_timeout("30")

    assert is_valid is True
    assert error is None


def test_validate_timeout_accepts_float() -> None:
    """Test timeout validation accepts float values."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_timeout("10.5")

    assert is_valid is True
    assert error is None


def test_validate_timeout_rejects_zero() -> None:
    """Test timeout validation rejects zero."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_timeout("0")

    assert is_valid is False
    assert "greater than 0" in error.lower()


def test_validate_timeout_rejects_negative() -> None:
    """Test timeout validation rejects negative values."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_timeout("-5")

    assert is_valid is False
    assert "greater than 0" in error.lower()


def test_validate_timeout_rejects_non_number() -> None:
    """Test timeout validation rejects non-numeric values."""
    builder = ToolFormBuilder()

    is_valid, error = builder._validate_timeout("abc")

    assert is_valid is False
    assert "positive number" in error.lower()


# =============================================================================
# Form creation tests
# =============================================================================


def test_create_alias_form_has_tool_alias_field() -> None:
    """Test _create_alias_form creates form with tool_alias field."""
    builder = ToolFormBuilder()

    form = builder._create_alias_form()

    assert form.get_field("tool_alias") is not None


def test_create_alias_form_uses_initial_data() -> None:
    """Test _create_alias_form populates default from initial_data."""
    builder = ToolFormBuilder()
    initial_data = {"tool_alias": "existing-alias"}

    form = builder._create_alias_form(initial_data)

    assert form.get_field("tool_alias").default == "existing-alias"


def test_create_optional_form_has_all_fields() -> None:
    """Test _create_optional_form creates form with all optional fields."""
    builder = ToolFormBuilder()

    form = builder._create_optional_form()

    assert form.get_field("allow_tools") is not None
    assert form.get_field("max_tool_call_turns") is not None
    assert form.get_field("timeout_sec") is not None


def test_create_optional_form_uses_initial_data() -> None:
    """Test _create_optional_form populates defaults from initial_data."""
    builder = ToolFormBuilder()
    initial_data = {
        "allow_tools": ["tool-a", "tool-b"],
        "max_tool_call_turns": 10,
        "timeout_sec": 60.0,
    }

    form = builder._create_optional_form(initial_data)

    assert form.get_field("allow_tools").default == "tool-a, tool-b"
    assert form.get_field("max_tool_call_turns").default == "10"
    assert form.get_field("timeout_sec").default == "60.0"


def test_create_optional_form_handles_empty_initial_data() -> None:
    """Test _create_optional_form handles empty initial_data gracefully."""
    builder = ToolFormBuilder()

    form = builder._create_optional_form({})

    # Defaults should be used
    assert form.get_field("allow_tools").default is None
    assert form.get_field("max_tool_call_turns").default == "5"
    assert form.get_field("timeout_sec").default is None


# =============================================================================
# Build config tests
# =============================================================================


def test_build_config_creates_tool_config() -> None:
    """Test _build_config creates valid ToolConfig."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1", "provider-2"],
        optional_data={"allow_tools": "", "max_tool_call_turns": "5", "timeout_sec": ""},
    )

    assert isinstance(config, ToolConfig)
    assert config.tool_alias == "my-tool"
    assert config.providers == ["provider-1", "provider-2"]
    assert config.max_tool_call_turns == 5


def test_build_config_parses_allow_tools() -> None:
    """Test _build_config parses comma-separated allow_tools."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={
            "allow_tools": "tool-a, tool-b , tool-c",
            "max_tool_call_turns": "5",
            "timeout_sec": "",
        },
    )

    assert config.allow_tools == ["tool-a", "tool-b", "tool-c"]


def test_build_config_handles_empty_allow_tools() -> None:
    """Test _build_config handles empty allow_tools."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={"allow_tools": "", "max_tool_call_turns": "5", "timeout_sec": ""},
    )

    assert config.allow_tools is None


def test_build_config_parses_max_tool_call_turns() -> None:
    """Test _build_config parses max_tool_call_turns as int."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={"allow_tools": "", "max_tool_call_turns": "15", "timeout_sec": ""},
    )

    assert config.max_tool_call_turns == 15


def test_build_config_defaults_max_tool_call_turns() -> None:
    """Test _build_config uses default max_tool_call_turns when empty."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={"allow_tools": "", "max_tool_call_turns": "", "timeout_sec": ""},
    )

    assert config.max_tool_call_turns == 5


def test_build_config_parses_timeout_sec() -> None:
    """Test _build_config parses timeout_sec as float."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={"allow_tools": "", "max_tool_call_turns": "5", "timeout_sec": "30.5"},
    )

    assert config.timeout_sec == 30.5


def test_build_config_handles_empty_timeout_sec() -> None:
    """Test _build_config handles empty timeout_sec (no timeout)."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={"allow_tools": "", "max_tool_call_turns": "5", "timeout_sec": ""},
    )

    assert config.timeout_sec is None


# =============================================================================
# Full run() workflow tests
# =============================================================================


@patch("data_designer.cli.forms.tool_builder.select_multiple_with_arrows")
@patch("data_designer.cli.forms.tool_builder.print_header")
@patch("data_designer.cli.forms.tool_builder.print_info")
@patch("data_designer.cli.forms.tool_builder.console")
def test_run_creates_tool_config(
    mock_console: MagicMock,
    mock_print_info: MagicMock,
    mock_print_header: MagicMock,
    mock_select_multiple: MagicMock,
) -> None:
    """Test run() creates valid ToolConfig."""
    builder = ToolFormBuilder(available_providers=["provider-1", "provider-2"])
    mock_select_multiple.return_value = ["provider-1"]

    mock_alias_form = MagicMock()
    mock_alias_form.prompt_all.return_value = {"tool_alias": "my-tool"}

    mock_optional_form = MagicMock()
    mock_optional_form.prompt_all.return_value = {
        "allow_tools": "",
        "max_tool_call_turns": "5",
        "timeout_sec": "",
    }

    with patch("data_designer.cli.forms.tool_builder.Form", side_effect=[mock_alias_form, mock_optional_form]):
        result = builder.run()

    assert isinstance(result, ToolConfig)
    assert result.tool_alias == "my-tool"
    assert result.providers == ["provider-1"]


@patch("data_designer.cli.forms.tool_builder.print_error")
@patch("data_designer.cli.forms.tool_builder.print_header")
@patch("data_designer.cli.forms.tool_builder.print_info")
def test_run_returns_none_when_no_providers(
    mock_print_info: MagicMock,
    mock_print_header: MagicMock,
    mock_print_error: MagicMock,
) -> None:
    """Test run() returns None when no providers available."""
    builder = ToolFormBuilder(available_providers=[])

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = {"tool_alias": "my-tool"}

    with patch("data_designer.cli.forms.tool_builder.Form", return_value=mock_form):
        result = builder.run()

    assert result is None
    mock_print_error.assert_called()


@patch("data_designer.cli.forms.tool_builder.confirm_action", return_value=True)
@patch("data_designer.cli.forms.tool_builder.print_header")
def test_run_returns_none_on_alias_cancel(
    mock_print_header: MagicMock,
    mock_confirm: MagicMock,
) -> None:
    """Test run() returns None when alias form is cancelled."""
    builder = ToolFormBuilder(available_providers=["provider-1"])

    mock_form = MagicMock()
    mock_form.prompt_all.return_value = None

    with patch("data_designer.cli.forms.tool_builder.Form", return_value=mock_form):
        result = builder.run()

    assert result is None


@patch("data_designer.cli.forms.tool_builder.select_multiple_with_arrows", return_value=[])
@patch("data_designer.cli.forms.tool_builder.print_error")
@patch("data_designer.cli.forms.tool_builder.print_header")
@patch("data_designer.cli.forms.tool_builder.print_info")
@patch("data_designer.cli.forms.tool_builder.console")
def test_run_shows_error_when_no_providers_selected(
    mock_console: MagicMock,
    mock_print_info: MagicMock,
    mock_print_header: MagicMock,
    mock_print_error: MagicMock,
    mock_select_multiple: MagicMock,
) -> None:
    """Test run() shows error when no providers are selected."""
    builder = ToolFormBuilder(available_providers=["provider-1"])

    mock_alias_form = MagicMock()
    # First call returns alias, second call (after error) cancels
    mock_alias_form.prompt_all.side_effect = [{"tool_alias": "my-tool"}, None]

    mock_confirm_action = MagicMock(return_value=True)

    with (
        patch("data_designer.cli.forms.tool_builder.Form", return_value=mock_alias_form),
        patch("data_designer.cli.forms.tool_builder.confirm_action", mock_confirm_action),
    ):
        result = builder.run()

    assert result is None
    mock_print_error.assert_called_with("At least one provider must be selected")


@patch("data_designer.cli.forms.tool_builder.confirm_action", return_value=True)
@patch("data_designer.cli.forms.tool_builder.select_multiple_with_arrows", return_value=None)
@patch("data_designer.cli.forms.tool_builder.print_header")
@patch("data_designer.cli.forms.tool_builder.print_info")
@patch("data_designer.cli.forms.tool_builder.console")
def test_run_handles_provider_selection_cancel(
    mock_console: MagicMock,
    mock_print_info: MagicMock,
    mock_print_header: MagicMock,
    mock_select_multiple: MagicMock,
    mock_confirm: MagicMock,
) -> None:
    """Test run() handles cancelled provider selection."""
    builder = ToolFormBuilder(available_providers=["provider-1"])

    mock_alias_form = MagicMock()
    mock_alias_form.prompt_all.side_effect = [{"tool_alias": "my-tool"}, None]

    with patch("data_designer.cli.forms.tool_builder.Form", return_value=mock_alias_form):
        result = builder.run()

    assert result is None


@patch("data_designer.cli.forms.tool_builder.select_multiple_with_arrows")
@patch("data_designer.cli.forms.tool_builder.print_header")
@patch("data_designer.cli.forms.tool_builder.print_info")
@patch("data_designer.cli.forms.tool_builder.console")
def test_run_uses_initial_data_for_providers(
    mock_console: MagicMock,
    mock_print_info: MagicMock,
    mock_print_header: MagicMock,
    mock_select_multiple: MagicMock,
) -> None:
    """Test run() passes initial provider selection from initial_data."""
    builder = ToolFormBuilder(available_providers=["provider-1", "provider-2"])
    initial_data = {
        "tool_alias": "existing-tool",
        "providers": ["provider-2"],
        "max_tool_call_turns": 10,
    }
    mock_select_multiple.return_value = ["provider-2"]

    mock_alias_form = MagicMock()
    mock_alias_form.prompt_all.return_value = {"tool_alias": "existing-tool"}

    mock_optional_form = MagicMock()
    mock_optional_form.prompt_all.return_value = {
        "allow_tools": "",
        "max_tool_call_turns": "10",
        "timeout_sec": "",
    }

    with patch("data_designer.cli.forms.tool_builder.Form", side_effect=[mock_alias_form, mock_optional_form]):
        builder.run(initial_data)

    # Check that select_multiple was called with default_keys from initial_data
    mock_select_multiple.assert_called_once()
    call_kwargs = mock_select_multiple.call_args[1]
    assert call_kwargs["default_keys"] == ["provider-2"]


@patch("data_designer.cli.forms.tool_builder.confirm_action")
@patch("data_designer.cli.forms.tool_builder.print_error")
@patch("data_designer.cli.forms.tool_builder.select_multiple_with_arrows")
@patch("data_designer.cli.forms.tool_builder.print_header")
@patch("data_designer.cli.forms.tool_builder.print_info")
@patch("data_designer.cli.forms.tool_builder.console")
def test_run_handles_build_config_exception(
    mock_console: MagicMock,
    mock_print_info: MagicMock,
    mock_print_header: MagicMock,
    mock_select_multiple: MagicMock,
    mock_print_error: MagicMock,
    mock_confirm: MagicMock,
) -> None:
    """Test run() handles exceptions during config building."""
    builder = ToolFormBuilder(available_providers=["provider-1"])
    mock_select_multiple.return_value = ["provider-1"]
    mock_confirm.side_effect = [False]  # Don't retry

    mock_alias_form = MagicMock()
    mock_alias_form.prompt_all.return_value = {"tool_alias": "my-tool"}

    mock_optional_form = MagicMock()
    mock_optional_form.prompt_all.return_value = {
        "allow_tools": "",
        "max_tool_call_turns": "5",
        "timeout_sec": "",
    }

    with (
        patch("data_designer.cli.forms.tool_builder.Form", side_effect=[mock_alias_form, mock_optional_form]),
        patch.object(builder, "_build_config", side_effect=Exception("Config error")),
    ):
        result = builder.run()

    assert result is None
    mock_print_error.assert_called_with("Configuration error: Config error")


# =============================================================================
# Edge cases and integration
# =============================================================================


def test_builder_title_is_set() -> None:
    """Test builder has appropriate title."""
    builder = ToolFormBuilder()
    assert builder.title == "Tool Configuration"


def test_existing_aliases_defaults_to_empty_set() -> None:
    """Test existing_aliases defaults to empty set when not provided."""
    builder = ToolFormBuilder()
    assert builder.existing_aliases == set()


def test_available_providers_defaults_to_empty_list() -> None:
    """Test available_providers defaults to empty list when not provided."""
    builder = ToolFormBuilder()
    assert builder.available_providers == []


def test_existing_aliases_is_stored() -> None:
    """Test existing_aliases is stored when provided."""
    aliases = {"tool-1", "tool-2"}
    builder = ToolFormBuilder(existing_aliases=aliases)
    assert builder.existing_aliases == aliases


def test_available_providers_is_stored() -> None:
    """Test available_providers is stored when provided."""
    providers = ["provider-1", "provider-2"]
    builder = ToolFormBuilder(available_providers=providers)
    assert builder.available_providers == providers


def test_build_config_strips_whitespace_from_allow_tools() -> None:
    """Test _build_config strips whitespace from allow_tools entries."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={
            "allow_tools": "  tool-a ,  tool-b  ",
            "max_tool_call_turns": "5",
            "timeout_sec": "",
        },
    )

    assert config.allow_tools == ["tool-a", "tool-b"]


def test_build_config_filters_empty_allow_tools_entries() -> None:
    """Test _build_config filters out empty entries from allow_tools."""
    builder = ToolFormBuilder()

    config = builder._build_config(
        tool_alias="my-tool",
        providers=["provider-1"],
        optional_data={
            "allow_tools": "tool-a, , ,tool-b, ",
            "max_tool_call_turns": "5",
            "timeout_sec": "",
        },
    )

    assert config.allow_tools == ["tool-a", "tool-b"]
