# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.cli.forms.field import TextField
from data_designer.cli.forms.form import Form
from data_designer.cli.ui import (
    confirm_action,
    console,
    print_error,
    print_header,
    print_info,
    select_multiple_with_arrows,
)
from data_designer.config.mcp import ToolConfig


class ToolFormBuilder:
    """Builds interactive forms for tool configuration.

    This builder uses a custom flow with multi-select for providers
    rather than the standard FormBuilder pattern.
    """

    def __init__(
        self,
        existing_aliases: set[str] | None = None,
        available_providers: list[str] | None = None,
    ):
        self.title = "Tool Configuration"
        self.existing_aliases = existing_aliases or set()
        self.available_providers = available_providers or []

    def run(self, initial_data: dict[str, Any] | None = None) -> ToolConfig | None:
        """Run the interactive tool configuration and return configured object."""
        print_header(self.title)

        while True:
            # Step 1: Get tool alias
            form = self._create_alias_form(initial_data)
            if initial_data:
                form.set_values(initial_data)

            result = form.prompt_all(allow_back=True)
            if result is None:
                if confirm_action("Cancel configuration?", default=False):
                    return None
                continue

            tool_alias = result["tool_alias"]

            # Step 2: Select providers (multi-select with checkboxes)
            if not self.available_providers:
                print_error("No MCP providers available. Please configure MCP providers first.")
                print_info("Run 'data-designer config mcp' to configure MCP providers.")
                return None

            console.print()
            print_info("Select one or more MCP providers for this tool configuration:")
            console.print()

            default_providers = initial_data.get("providers") if initial_data else None
            provider_options = {p: p for p in self.available_providers}
            selected_providers = select_multiple_with_arrows(
                provider_options,
                "Select MCP providers (Space to toggle, Enter to confirm):",
                default_keys=default_providers,
                allow_empty=False,
            )

            if selected_providers is None:
                if confirm_action("Cancel configuration?", default=False):
                    return None
                continue

            if not selected_providers:
                print_error("At least one provider must be selected")
                continue

            # Step 3: Get optional settings
            optional_form = self._create_optional_form(initial_data)
            if initial_data:
                optional_form.set_values(initial_data)

            optional_result = optional_form.prompt_all(allow_back=True)
            if optional_result is None:
                continue  # Go back to start

            try:
                config = self._build_config(tool_alias, selected_providers, optional_result)
                return config
            except Exception as e:
                print_error(f"Configuration error: {e}")
                if not confirm_action("Try again?", default=True):
                    return None

    def _create_alias_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the form for tool alias."""
        fields = [
            TextField(
                "tool_alias",
                "Tool alias (used to reference this config in columns)",
                default=initial_data.get("tool_alias") if initial_data else None,
                required=True,
                validator=self._validate_alias,
            ),
        ]
        return Form("Tool Alias", fields)

    def _create_optional_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the form for optional tool settings."""
        # Convert allow_tools list to comma-separated string for display
        allow_tools_default = None
        if initial_data and initial_data.get("allow_tools"):
            allow_tools_default = ", ".join(initial_data["allow_tools"])

        fields = [
            TextField(
                "allow_tools",
                "Allowed tools (comma-separated, leave empty for all)",
                default=allow_tools_default,
                required=False,
            ),
            TextField(
                "max_tool_call_turns",
                "Max tool-calling turns (a turn may execute multiple parallel tools)",
                default=str(initial_data.get("max_tool_call_turns", 5)) if initial_data else "5",
                required=False,
                validator=self._validate_max_tool_call_turns,
            ),
            TextField(
                "timeout_sec",
                "Timeout in seconds per tool call (leave empty for no timeout)",
                default=str(initial_data.get("timeout_sec", "")) if initial_data else None,
                required=False,
                validator=self._validate_timeout,
            ),
        ]
        return Form("Optional Settings", fields)

    def _validate_alias(self, alias: str) -> tuple[bool, str | None]:
        """Validate tool alias."""
        if not alias:
            return False, "Tool alias is required"
        if alias in self.existing_aliases:
            return False, f"Tool alias '{alias}' already exists"
        return True, None

    def _validate_max_tool_call_turns(self, value: str) -> tuple[bool, str | None]:
        """Validate max_tool_call_turns."""
        if not value:
            return True, None  # Will use default
        try:
            int_value = int(value)
            if int_value < 1:
                return False, "Must be at least 1"
            return True, None
        except ValueError:
            return False, "Must be a positive integer"

    def _validate_timeout(self, value: str) -> tuple[bool, str | None]:
        """Validate timeout_sec."""
        if not value:
            return True, None  # No timeout is valid
        try:
            float_value = float(value)
            if float_value <= 0:
                return False, "Must be greater than 0"
            return True, None
        except ValueError:
            return False, "Must be a positive number"

    def _build_config(
        self,
        tool_alias: str,
        providers: list[str],
        optional_data: dict[str, Any],
    ) -> ToolConfig:
        """Build ToolConfig from collected data."""
        # Parse allow_tools from comma-separated string
        allow_tools = None
        if optional_data.get("allow_tools"):
            allow_tools = [t.strip() for t in optional_data["allow_tools"].split(",") if t.strip()]

        # Parse max_tool_call_turns
        max_tool_call_turns = 5
        if optional_data.get("max_tool_call_turns"):
            max_tool_call_turns = int(optional_data["max_tool_call_turns"])

        # Parse timeout_sec
        timeout_sec = None
        if optional_data.get("timeout_sec"):
            timeout_sec = float(optional_data["timeout_sec"])

        return ToolConfig(
            tool_alias=tool_alias,
            providers=providers,
            allow_tools=allow_tools if allow_tools else None,
            max_tool_call_turns=max_tool_call_turns,
            timeout_sec=timeout_sec,
        )
