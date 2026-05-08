# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from collections.abc import Callable

from packaging.utils import canonicalize_name

from data_designer.cli.plugin_catalog import (
    PLUGIN_ENTRY_POINT_GROUP,
    PYPI_SIMPLE_INDEX_URL,
    InstallPlan,
    PluginCatalogConfig,
    PluginCatalogEntry,
    UninstallPlan,
)

InstallRunner = Callable[[list[str]], int]
PIP_EXTRA_INDEX_SOURCE_WARNING = (
    "pip --extra-index-url is not source-pinned; pip may choose a same-named package from another configured index. "
    "Use uv or a direct reference when strict source selection is required."
)


class PluginInstallService:
    """Resolve, execute, and verify plugin package install/uninstall plans."""

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
        install_args, source_description, source_warning = _install_args_for_entry(entry, resolved_manager)
        command = _base_command(resolved_manager) + install_args
        return InstallPlan(
            package_name=entry.package.name,
            source_description=source_description,
            command=command,
            manager=resolved_manager,
            catalog_alias=catalog.alias,
            trusted_catalog=catalog.trusted,
            source_warning=source_warning,
        )

    def build_uninstall_plan(
        self,
        entry: PluginCatalogEntry,
        catalog: PluginCatalogConfig,
        *,
        manager: str = "auto",
    ) -> UninstallPlan:
        """Build the exact package-manager command to uninstall one catalog package."""
        resolved_manager = _resolve_manager(manager)
        return UninstallPlan(
            package_name=entry.package.name,
            command=_base_uninstall_command(resolved_manager) + [entry.package.name],
            manager=resolved_manager,
            catalog_alias=catalog.alias,
        )

    def install(self, plan: InstallPlan) -> None:
        """Run an installation plan.

        Raises:
            RuntimeError: If the package manager exits unsuccessfully.
        """
        return_code = self._runner(plan.command)
        if return_code != 0:
            raise RuntimeError(f"Plugin package installer exited with status {return_code}")

    def uninstall(self, plan: UninstallPlan) -> None:
        """Run an uninstall plan.

        Raises:
            RuntimeError: If the package manager exits unsuccessfully.
        """
        return_code = self._runner(plan.command)
        if return_code != 0:
            raise RuntimeError(f"Plugin package uninstaller exited with status {return_code}")

    def verify_entry_point(self, entry: PluginCatalogEntry) -> bool:
        """Verify the plugin's declared entry point is installed."""
        return self.verify_entry_points([entry])

    def verify_entry_points(self, entries: list[PluginCatalogEntry]) -> bool:
        """Verify every declared entry point for an installed catalog package."""
        if not entries:
            return False

        importlib.invalidate_caches()
        installed_entry_points = list(importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP))
        return all(
            any(
                _installed_entry_point_matches(installed_entry_point, entry)
                for installed_entry_point in installed_entry_points
            )
            for entry in entries
        )

    def verify_entry_points_removed(self, entries: list[PluginCatalogEntry]) -> bool:
        """Verify every declared entry point for a catalog package is no longer installed."""
        if not entries:
            return False

        importlib.invalidate_caches()
        installed_entry_points = list(importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP))
        return all(
            not any(
                _installed_entry_point_matches(installed_entry_point, entry)
                for installed_entry_point in installed_entry_points
            )
            for entry in entries
        )


def _run_subprocess(command: list[str]) -> int:
    result = subprocess.run(command, check=False, stdin=subprocess.DEVNULL)
    return result.returncode


def _installed_entry_point_matches(
    installed_entry_point: importlib.metadata.EntryPoint,
    entry: PluginCatalogEntry,
) -> bool:
    if installed_entry_point.name != entry.entry_point.name:
        return False
    if installed_entry_point.value != entry.entry_point.value:
        return False

    distribution_name = _entry_point_distribution_name(installed_entry_point)
    if distribution_name is None:
        return True
    return canonicalize_name(distribution_name) == canonicalize_name(entry.package.name)


def _entry_point_distribution_name(installed_entry_point: importlib.metadata.EntryPoint) -> str | None:
    distribution = getattr(installed_entry_point, "dist", None)
    if distribution is None:
        return None

    metadata = getattr(distribution, "metadata", None)
    if metadata is None:
        return None

    name = metadata.get("Name")
    if not isinstance(name, str) or not name:
        return None
    return name


def _resolve_manager(manager: str) -> str:
    if manager not in {"auto", "uv", "pip"}:
        raise ValueError(f"Unsupported plugin installer {manager!r}. Expected 'auto', 'uv', or 'pip'.")
    if manager == "auto":
        return "uv" if shutil.which("uv") else "pip"
    if manager == "uv" and not shutil.which("uv"):
        raise ValueError("uv was requested for plugin package installation, but it is not available on PATH")
    return manager


def _base_command(manager: str) -> list[str]:
    if manager == "uv":
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _base_uninstall_command(manager: str) -> list[str]:
    if manager == "uv":
        return ["uv", "pip", "uninstall", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "uninstall", "--yes"]


def _install_args_for_entry(entry: PluginCatalogEntry, manager: str) -> tuple[list[str], str, str | None]:
    requirement = entry.install.requirement
    index_url = entry.install.index_url
    if index_url is None:
        return [requirement], requirement, None

    if manager == "uv":
        return (
            ["--default-index", PYPI_SIMPLE_INDEX_URL, "--index", index_url, requirement],
            f"{requirement} via {index_url}",
            None,
        )
    return (
        ["--extra-index-url", index_url, requirement],
        f"{requirement} via {index_url}",
        PIP_EXTRA_INDEX_SOURCE_WARNING,
    )
