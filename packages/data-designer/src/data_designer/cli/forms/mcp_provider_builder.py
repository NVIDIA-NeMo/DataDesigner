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
    select_with_arrows,
)
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, MCPProviderT
from data_designer.config.utils.io_helpers import is_http_url


class MCPProviderFormBuilder:
    """Builds interactive forms for MCP provider configuration.

    Supports both MCPProvider (remote SSE) and LocalStdioMCPProvider (subprocess).
    """

    def __init__(self, existing_names: set[str] | None = None):
        self.title = "MCP Provider Configuration"
        self.existing_names = existing_names or set()

    def run(self, initial_data: dict[str, Any] | None = None) -> MCPProviderT | None:
        """Run the interactive MCP provider configuration and return configured object."""
        print_header(self.title)

        while True:
            # Determine provider type
            if initial_data and initial_data.get("provider_type"):
                provider_type = initial_data["provider_type"]
            else:
                provider_type = self._select_provider_type()
                if provider_type is None:
                    return None

            # Run appropriate form based on provider type
            if provider_type == "sse":
                result = self._run_sse_form(initial_data)
            else:  # stdio
                result = self._run_stdio_form(initial_data)

            if result is not None:
                return result

            # If form was cancelled, ask if they want to try again
            if not confirm_action("Try a different provider type?", default=False):
                return None
            initial_data = None  # Reset for new selection

    def _select_provider_type(self) -> str | None:
        """Prompt user to select provider type."""
        options = {
            "sse": "Remote SSE server (connect to existing server)",
            "stdio": "Local stdio subprocess (launch server as subprocess)",
        }

        console.print()
        return select_with_arrows(
            options,
            "What type of MCP provider?",
            default_key="sse",
            allow_back=True,
        )

    def _run_sse_form(self, initial_data: dict[str, Any] | None = None) -> MCPProvider | None:
        """Run form for remote SSE provider."""
        fields = [
            TextField(
                "name",
                "MCP provider name",
                default=initial_data.get("name") if initial_data else None,
                required=True,
                validator=self._validate_name,
            ),
            TextField(
                "endpoint",
                "SSE endpoint URL",
                default=initial_data.get("endpoint") if initial_data else None,
                required=True,
                validator=self._validate_endpoint,
            ),
            TextField(
                "api_key",
                "API key or environment variable name (optional)",
                default=initial_data.get("api_key") if initial_data else None,
                required=False,
            ),
        ]

        form = Form("Remote SSE Provider", fields)
        if initial_data:
            form.set_values(initial_data)

        result = form.prompt_all(allow_back=True)
        if result is None:
            return None

        try:
            return MCPProvider(
                name=result["name"],
                endpoint=result["endpoint"],
                api_key=result.get("api_key") or None,
            )
        except Exception as e:
            print_error(f"Configuration error: {e}")
            return None

    def _run_stdio_form(self, initial_data: dict[str, Any] | None = None) -> LocalStdioMCPProvider | None:
        """Run form for local stdio provider."""
        # Convert args list to comma-separated string for display
        args_default = None
        if initial_data and initial_data.get("args"):
            args_default = ",".join(initial_data["args"])

        # Convert env dict to KEY=VALUE,KEY2=VALUE2 format for display
        env_default = None
        if initial_data and initial_data.get("env"):
            env_default = ",".join(f"{k}={v}" for k, v in initial_data["env"].items())

        fields = [
            TextField(
                "name",
                "MCP provider name",
                default=initial_data.get("name") if initial_data else None,
                required=True,
                validator=self._validate_name,
            ),
            TextField(
                "command",
                "Command to run (e.g., python, node, npx)",
                default=initial_data.get("command") if initial_data else None,
                required=True,
                validator=self._validate_command,
            ),
            TextField(
                "args",
                "Arguments (comma-separated, e.g., -m,my_server,--port,8080)",
                default=args_default,
                required=False,
            ),
            TextField(
                "env",
                "Environment variables (KEY=VALUE, comma-separated)",
                default=env_default,
                required=False,
            ),
        ]

        form = Form("Local Stdio Provider", fields)
        if initial_data:
            form.set_values(
                {
                    "name": initial_data.get("name"),
                    "command": initial_data.get("command"),
                    "args": args_default,
                    "env": env_default,
                }
            )

        result = form.prompt_all(allow_back=True)
        if result is None:
            return None

        try:
            # Parse args from comma-separated string
            args: list[str] = []
            if result.get("args"):
                args = [a.strip() for a in result["args"].split(",") if a.strip()]

            # Parse env from KEY=VALUE format
            env: dict[str, str] = {}
            if result.get("env"):
                for pair in result["env"].split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        env[key.strip()] = value.strip()

            return LocalStdioMCPProvider(
                name=result["name"],
                command=result["command"],
                args=args,
                env=env,
            )
        except Exception as e:
            print_error(f"Configuration error: {e}")
            return None

    def _validate_name(self, name: str) -> tuple[bool, str | None]:
        """Validate MCP provider name."""
        if not name:
            return False, "MCP provider name is required"
        if name in self.existing_names:
            return False, f"MCP provider '{name}' already exists"
        return True, None

    def _validate_endpoint(self, endpoint: str) -> tuple[bool, str | None]:
        """Validate endpoint URL."""
        if not endpoint:
            return False, "Endpoint URL is required"
        if not is_http_url(endpoint):
            return False, "Invalid URL format (must start with http:// or https://)"
        return True, None

    def _validate_command(self, command: str) -> tuple[bool, str | None]:
        """Validate command."""
        if not command:
            return False, "Command is required"
        if not command.strip():
            return False, "Command cannot be empty"
        return True, None
