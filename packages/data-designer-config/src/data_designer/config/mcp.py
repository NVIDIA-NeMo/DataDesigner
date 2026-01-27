# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.errors import InvalidConfigError


class MCPServerConfig(ConfigBase):
    """Configuration for a single MCP server connection.

    MCP servers can be launched locally via stdio (command/args) or accessed remotely
    over SSE (url), allowing the same configuration type to cover both deployment modes.

    Attributes:
        name (str): Unique name used to reference this MCP server.
        command (str | None): Executable to launch the MCP server via stdio transport. Defaults to None.
        args (list[str]): Arguments passed to the MCP server executable. Defaults to [].
        url (str | None): SSE endpoint URL for connecting to a remote MCP server. Defaults to None.
        env (dict[str, str]): Environment variables passed to the MCP server subprocess. Defaults to {}.

    Examples:
        Stdio (subprocess) transport:

        >>> MCPServerConfig(
        ...     name="demo-mcp",
        ...     command="python",
        ...     args=["-m", "data_designer_e2e_tests.mcp_demo_server"],
        ...     env={"PYTHONPATH": "/path/to/project"},
        ... )

        SSE (HTTP) transport:

        >>> MCPServerConfig(
        ...     name="remote-mcp",
        ...     url="http://localhost:8080/sse",
        ... )
    """

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transport(self) -> Self:
        """Validate that exactly one transport is configured.

        Returns:
            The validated MCPServerConfig instance.

        Raises:
            InvalidConfigError: If both or neither of `command` and `url` are provided,
                or if `args` are supplied for an SSE-based server.
        """
        if bool(self.command) == bool(self.url):
            raise InvalidConfigError("MCP server config must define exactly one of 'command' or 'url'.")
        if self.url and self.args:
            raise InvalidConfigError("MCP server config 'args' is only valid when using 'command'.")
        return self


class MCPToolConfig(ConfigBase):
    """Configuration for permitting MCP tools on an LLM column.

    Attributes:
        server_name (str): Name of the MCP server to use for tool calls.
        tool_names (list[str] | None): Optional allowlist of tool names. If None, all tools are allowed. Defaults to None.
        max_tool_calls (int): Maximum number of tool calls permitted in a single generation. Defaults to 5.
        timeout_sec (float | None): Timeout in seconds for MCP tool calls. Defaults to None (no timeout).
    """

    server_name: str
    tool_names: list[str] | None = None
    max_tool_calls: int = Field(default=5, ge=1)
    timeout_sec: float | None = Field(default=None, gt=0)
