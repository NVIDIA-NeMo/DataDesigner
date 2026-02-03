# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field
from typing_extensions import TypeAlias

from data_designer.base import ConfigBase


class MCPProvider(ConfigBase):
    """Configuration for a remote MCP server connection.

    MCPProvider is used to connect to pre-existing MCP servers via SSE (Server-Sent Events)
    transport. For local subprocess-based MCP servers, use LocalStdioMCPProvider instead.

    Attributes:
        name (str): Unique name used to reference this MCP provider.
        endpoint (str): SSE endpoint URL for connecting to the remote MCP server.
        api_key (str | None): Optional API key for authentication. Defaults to None.
        provider_type (Literal["sse"]): Transport type discriminator, always "sse".

    Examples:
        Remote SSE transport:

        >>> MCPProvider(
        ...     name="remote-mcp",
        ...     endpoint="http://localhost:8080/sse",
        ...     api_key="your-api-key",
        ... )
    """

    provider_type: Literal["sse"] = "sse"
    name: str
    endpoint: str
    api_key: str | None = None


class LocalStdioMCPProvider(ConfigBase):
    """Configuration for launching a local MCP server via stdio transport.

    LocalStdioMCPProvider is used to launch MCP servers as subprocesses using stdio
    for communication. For connecting to remote/pre-existing MCP servers, use MCPProvider instead.

    Attributes:
        name (str): Unique name used to reference this MCP provider.
        command (str): Executable to launch the MCP server via stdio transport.
        args (list[str]): Arguments passed to the MCP server executable. Defaults to [].
        env (dict[str, str]): Environment variables passed to the MCP server subprocess. Defaults to {}.
        provider_type (Literal["stdio"]): Transport type discriminator, always "stdio".

    Examples:
        Stdio (subprocess) transport:

        >>> LocalStdioMCPProvider(
        ...     name="demo-mcp",
        ...     command="python",
        ...     args=["-m", "data_designer_e2e_tests.mcp_demo_server"],
        ...     env={"PYTHONPATH": "/path/to/project"},
        ... )
    """

    provider_type: Literal["stdio"] = "stdio"
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


MCPProviderT: TypeAlias = Annotated[MCPProvider | LocalStdioMCPProvider, Field(discriminator="provider_type")]


class ToolConfig(ConfigBase):
    """Configuration for permitting MCP tools on an LLM column.

    ToolConfig defines which tools are available for use during LLM generation.
    It references one or more MCP providers by name and can optionally restrict
    which tools from those providers are permitted.

    Attributes:
        tool_alias (str): User-defined alias to reference this tool configuration in column configs.
        providers (list[str]): Names of the MCP providers to use for tool calls. Tools can be
            drawn from multiple providers.
        allow_tools (list[str] | None): Optional allowlist of tool names that restricts which
            tools are permitted. If None, all tools from the specified providers are allowed.
            Defaults to None.
        max_tool_call_turns (int): Maximum number of tool-calling turns permitted in a single
            generation. A turn is one iteration where the LLM requests tool calls. With parallel
            tool calling, a single turn may execute multiple tools simultaneously. Defaults to 5.
        timeout_sec (float | None): Timeout in seconds for MCP tool calls. Defaults to None (no timeout).

    Examples:
        >>> ToolConfig(
        ...     tool_alias="search-tools",
        ...     providers=["doc-search-mcp", "web-search-mcp"],
        ...     allow_tools=["search_docs", "list_docs"],
        ...     max_tool_call_turns=10,
        ...     timeout_sec=30.0,
        ... )
    """

    tool_alias: str
    providers: list[str]
    allow_tools: list[str] | None = None
    max_tool_call_turns: int = Field(default=5, ge=1)
    timeout_sec: float | None = Field(default=None, gt=0)
