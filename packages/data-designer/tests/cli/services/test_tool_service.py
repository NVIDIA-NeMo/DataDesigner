# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_designer.cli.repositories.tool_repository import ToolRepository
from data_designer.cli.services.tool_service import ToolService
from data_designer.config.mcp import ToolConfig


def test_list_all(stub_tool_service: ToolService, stub_tool_configs: list[ToolConfig]) -> None:
    assert stub_tool_service.list_all() == stub_tool_configs


def test_get_by_alias(stub_tool_service: ToolService, stub_tool_configs: list[ToolConfig]) -> None:
    assert stub_tool_service.get_by_alias("tool-config-1") == stub_tool_configs[0]
    assert stub_tool_service.get_by_alias("tool-config-3") is None


def test_add(
    stub_tool_service: ToolService,
    stub_tool_configs: list[ToolConfig],
    stub_new_tool_config: ToolConfig,
) -> None:
    stub_tool_service.add(stub_new_tool_config)
    assert stub_tool_service.list_all() == stub_tool_configs + [stub_new_tool_config]


def test_add_duplicate_alias(stub_tool_service: ToolService) -> None:
    """Test adding a tool config with an alias that already exists."""
    duplicate_config = ToolConfig(
        tool_alias="tool-config-1",
        providers=["mcp-provider-1"],
    )
    with pytest.raises(ValueError, match="Tool config with alias 'tool-config-1' already exists"):
        stub_tool_service.add(duplicate_config)


def test_update(stub_tool_service: ToolService, stub_new_tool_config: ToolConfig) -> None:
    stub_tool_service.update("tool-config-1", stub_new_tool_config)
    assert stub_tool_service.get_by_alias("tool-config-1") is None
    assert stub_tool_service.get_by_alias("tool-config-3") == stub_new_tool_config


def test_update_no_registry(tmp_path: Path, stub_new_tool_config: ToolConfig) -> None:
    """Test updating when no registry exists."""
    service = ToolService(ToolRepository(tmp_path))
    with pytest.raises(ValueError, match="No tool configs configured"):
        service.update("tool-config-1", stub_new_tool_config)


def test_update_nonexistent_config(stub_tool_service: ToolService, stub_new_tool_config: ToolConfig) -> None:
    """Test updating a tool config that doesn't exist."""
    with pytest.raises(ValueError, match="Tool config with alias 'nonexistent' not found"):
        stub_tool_service.update("nonexistent", stub_new_tool_config)


def test_update_to_existing_alias(stub_tool_service: ToolService, stub_tool_configs: list[ToolConfig]) -> None:
    """Test updating a tool config to an alias that already exists."""
    updated_config = ToolConfig(
        tool_alias="tool-config-2",  # Already exists
        providers=["mcp-provider-1"],
    )
    with pytest.raises(ValueError, match="Tool config alias 'tool-config-2' already exists"):
        stub_tool_service.update("tool-config-1", updated_config)


def test_delete(stub_tool_service: ToolService) -> None:
    stub_tool_service.delete("tool-config-1")
    assert len(stub_tool_service.list_all()) == 1


def test_delete_no_registry(tmp_path: Path) -> None:
    """Test deleting when no registry exists."""
    service = ToolService(ToolRepository(tmp_path))
    with pytest.raises(ValueError, match="No tool configs configured"):
        service.delete("tool-config-1")


def test_delete_nonexistent_config(stub_tool_service: ToolService) -> None:
    """Test deleting a tool config that doesn't exist."""
    with pytest.raises(ValueError, match="Tool config with alias 'nonexistent' not found"):
        stub_tool_service.delete("nonexistent")


def test_delete_last_config(stub_tool_service: ToolService, stub_tool_configs: list[ToolConfig]) -> None:
    """Test deleting all tool configs triggers repository.delete()."""
    stub_tool_service.delete("tool-config-1")
    stub_tool_service.delete("tool-config-2")
    assert stub_tool_service.list_all() == []
