# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from collections.abc import Callable

from data_designer.cli.plugin_catalog import (
    PLUGIN_ENTRY_POINT_GROUP,
    PYPI_SIMPLE_INDEX_URL,
    InstallPlan,
    PluginCatalogConfig,
    PluginCatalogEntry,
)

InstallRunner = Callable[[list[str]], int]


class PluginInstallService:
    """Resolve, execute, and verify plugin installation plans."""

    def __init__(self, runner: InstallRunner | None = None) -> None:
        self._runner = runner or _run_subprocess

    def build_install_plan(
        self,
        entry: PluginCatalogEntry,
        catalog: PluginCatalogConfig,
        *,
        manager: str = "auto",
    ) -> InstallPlan:
        """Build the exact package-manager command for one catalog entry."""
        resolved_manager = _resolve_manager(manager)
        install_args, source_description = _install_args_for_entry(entry, resolved_manager)
        command = _base_command(resolved_manager) + install_args
        return InstallPlan(
            plugin_name=entry.name,
            package_name=entry.package.name,
            source_description=source_description,
            command=command,
            manager=resolved_manager,
            catalog_alias=catalog.alias,
            trusted_catalog=catalog.trusted,
        )

    def install(self, plan: InstallPlan) -> None:
        """Run an installation plan.

        Raises:
            RuntimeError: If the package manager exits unsuccessfully.
        """
        return_code = self._runner(plan.command)
        if return_code != 0:
            raise RuntimeError(f"Plugin installer exited with status {return_code}")

    def verify_entry_point(self, entry: PluginCatalogEntry) -> bool:
        """Verify the plugin's declared entry point is installed."""
        return self.verify_entry_points([entry])

    def verify_entry_points(self, entries: list[PluginCatalogEntry]) -> bool:
        """Verify every declared entry point for an installed catalog package."""
        importlib.invalidate_caches()
        installed_entry_point_names = {
            entry_point.name for entry_point in importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        }
        return bool(entries) and all(entry.entry_point.name in installed_entry_point_names for entry in entries)


def _run_subprocess(command: list[str]) -> int:
    result = subprocess.run(command, check=False)
    return result.returncode


def _resolve_manager(manager: str) -> str:
    if manager not in {"auto", "uv", "pip"}:
        raise ValueError(f"Unsupported plugin installer {manager!r}. Expected 'auto', 'uv', or 'pip'.")
    if manager == "auto":
        return "uv" if shutil.which("uv") else "pip"
    if manager == "uv" and not shutil.which("uv"):
        raise ValueError("uv was requested for plugin installation, but it is not available on PATH")
    return manager


def _base_command(manager: str) -> list[str]:
    if manager == "uv":
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _install_args_for_entry(entry: PluginCatalogEntry, manager: str) -> tuple[list[str], str]:
    requirement = entry.install.requirement
    index_url = entry.install.index_url
    if index_url is None:
        return [requirement], requirement

    if manager == "uv":
        return (
            ["--default-index", PYPI_SIMPLE_INDEX_URL, "--index", index_url, requirement],
            f"{requirement} via {index_url}",
        )
    return (
        ["--extra-index-url", index_url, requirement],
        f"{requirement} via {index_url}",
    )
