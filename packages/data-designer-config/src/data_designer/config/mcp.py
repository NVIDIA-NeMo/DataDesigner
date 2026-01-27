# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.errors import InvalidConfigError


class MCPServerConfig(ConfigBase):
    """Configuration for a single MCP server connection."""

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transport(self) -> Self:
        if bool(self.command) == bool(self.url):
            raise InvalidConfigError("MCP server config must define exactly one of 'command' or 'url'.")
        if self.url and self.args:
            raise InvalidConfigError("MCP server config 'args' is only valid when using 'command'.")
        return self


class MCPToolConfig(ConfigBase):
    """Configuration for permitting MCP tools on an LLM column."""

    server_name: str
    tool_names: list[str] | None = None
    max_tool_calls: int = Field(default=5, ge=1)
