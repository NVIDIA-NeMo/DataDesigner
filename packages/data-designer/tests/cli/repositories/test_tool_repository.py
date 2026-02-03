# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.cli.repositories.tool_repository import ToolConfigRegistry, ToolRepository
from data_designer.config.mcp import ToolConfig
from data_designer.config.utils.constants import TOOL_CONFIGS_FILE_NAME
from data_designer.config.utils.io_helpers import save_config_file


def test_config_file(tmp_path: Path) -> None:
    repository = ToolRepository(tmp_path)
    assert repository.config_file == tmp_path / TOOL_CONFIGS_FILE_NAME


def test_load_does_not_exist() -> None:
    repository = ToolRepository(Path("non_existent_path"))
    assert repository.load() is None


def test_load_exists(tmp_path: Path, stub_tool_configs: list[ToolConfig]) -> None:
    config_file_path = tmp_path / TOOL_CONFIGS_FILE_NAME
    save_config_file(
        config_file_path,
        ToolConfigRegistry(tool_configs=stub_tool_configs).model_dump(),
    )
    repository = ToolRepository(tmp_path)
    assert repository.load() is not None
    assert repository.load().tool_configs == stub_tool_configs


def test_save(tmp_path: Path, stub_tool_configs: list[ToolConfig]) -> None:
    repository = ToolRepository(tmp_path)
    repository.save(ToolConfigRegistry(tool_configs=stub_tool_configs))
    assert repository.load() is not None
    assert repository.load().tool_configs == stub_tool_configs
