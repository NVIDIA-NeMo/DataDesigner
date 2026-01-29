# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from data_designer.cli.repositories.base import ConfigRepository
from data_designer.config.mcp import MCPProviderT
from data_designer.config.utils.constants import MCP_PROVIDERS_FILE_NAME
from data_designer.config.utils.io_helpers import load_config_file, save_config_file


class MCPProviderRegistry(BaseModel):
    """Registry for MCP provider configurations.

    Supports both MCPProvider (remote SSE) and LocalStdioMCPProvider (subprocess)
    via the discriminated union MCPProviderT.
    """

    providers: list[MCPProviderT]


class MCPProviderRepository(ConfigRepository[MCPProviderRegistry]):
    """Repository for MCP provider configurations."""

    @property
    def config_file(self) -> Path:
        """Get the MCP provider configuration file path."""
        return self.config_dir / MCP_PROVIDERS_FILE_NAME

    def load(self) -> MCPProviderRegistry | None:
        """Load MCP provider configuration from file."""
        if not self.exists():
            return None

        try:
            config_dict = load_config_file(self.config_file)
            return MCPProviderRegistry.model_validate(config_dict)
        except Exception:
            return None

    def save(self, config: MCPProviderRegistry) -> None:
        """Save MCP provider configuration to file."""
        config_dict = config.model_dump(mode="json", exclude_none=True)
        save_config_file(self.config_file, config_dict)
