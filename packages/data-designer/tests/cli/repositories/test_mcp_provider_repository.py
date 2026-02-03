# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRegistry, MCPProviderRepository
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider
from data_designer.config.utils.constants import MCP_PROVIDERS_FILE_NAME
from data_designer.config.utils.io_helpers import save_config_file

MCPProviderT = MCPProvider | LocalStdioMCPProvider


def test_config_file(tmp_path: Path) -> None:
    repository = MCPProviderRepository(tmp_path)
    assert repository.config_file == tmp_path / MCP_PROVIDERS_FILE_NAME


def test_load_does_not_exist() -> None:
    repository = MCPProviderRepository(Path("non_existent_path"))
    assert repository.load() is None


def test_load_exists(tmp_path: Path, stub_mcp_providers: list[MCPProviderT]) -> None:
    providers_file_path = tmp_path / MCP_PROVIDERS_FILE_NAME
    save_config_file(
        providers_file_path,
        MCPProviderRegistry(providers=stub_mcp_providers).model_dump(),
    )
    repository = MCPProviderRepository(tmp_path)
    assert repository.load() is not None
    assert repository.load().providers == stub_mcp_providers


def test_save(tmp_path: Path, stub_mcp_providers: list[MCPProviderT]) -> None:
    repository = MCPProviderRepository(tmp_path)
    repository.save(MCPProviderRegistry(providers=stub_mcp_providers))
    assert repository.load() is not None
    assert repository.load().providers == stub_mcp_providers
