# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from data_designer.cli.plugin_catalog import InstallPlan, PluginCatalogEntry, PluginSourceInfo, PluginTapConfig
from data_designer.plugins.registry import PluginRegistry

InstallRunner = Callable[[list[str]], int]


class PluginInstallService:
    """Resolve, execute, and verify plugin installation plans."""

    def __init__(self, runner: InstallRunner | None = None) -> None:
        self._runner = runner or _run_subprocess

    def build_install_plan(
        self,
        entry: PluginCatalogEntry,
        tap: PluginTapConfig,
        *,
        manager: str = "auto",
    ) -> InstallPlan:
        """Build the exact package-manager command for one catalog entry."""
        resolved_manager = _resolve_manager(manager)
        install_args, source_description = _install_args_for_entry(entry, tap)
        command = _base_command(resolved_manager) + install_args
        return InstallPlan(
            plugin_name=entry.name,
            package_name=entry.package.name,
            source_description=source_description,
            command=command,
            manager=resolved_manager,
            tap_alias=tap.alias,
            trusted_tap=tap.trusted,
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
        """Verify the plugin is discoverable by the runtime PluginRegistry."""
        PluginRegistry.reset()
        registry = PluginRegistry()
        return registry.plugin_exists(entry.name)


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


def _install_args_for_entry(entry: PluginCatalogEntry, tap: PluginTapConfig) -> tuple[list[str], str]:
    source = entry.source
    if source is None:
        raise ValueError(
            f"Plugin {entry.name!r} cannot be installed because the catalog entry does not declare a source"
        )

    source_type = source.type.lower()
    if source_type == "pypi":
        target = _pypi_target(entry, source)
        return [target], target
    if source_type == "git":
        target = _git_target(source)
        return [target], target
    if source_type == "path":
        args = _path_args(entry, source, tap)
        return args, " ".join(args)
    if source_type == "url":
        target = _required(source.url, "url", source_type)
        return [target], target

    raise ValueError(f"Plugin {entry.name!r} declares unsupported install source type {source.type!r}")


def _pypi_target(entry: PluginCatalogEntry, source: PluginSourceInfo) -> str:
    package_name = source.package or entry.package.name
    if entry.package.version:
        return f"{package_name}=={entry.package.version}"
    return package_name


def _git_target(source: PluginSourceInfo) -> str:
    url = _required(source.url, "url", "git")
    target = url if url.startswith("git+") else f"git+{url}"
    if source.ref:
        target = f"{target}@{source.ref}"

    fragments = []
    if source.subdirectory:
        fragments.append(f"subdirectory={source.subdirectory}")
    if fragments:
        target = f"{target}#{'&'.join(fragments)}"
    return target


def _path_args(entry: PluginCatalogEntry, source: PluginSourceInfo, tap: PluginTapConfig) -> list[str]:
    path = source.path or entry.package.path
    if path is None:
        raise ValueError(f"Plugin {entry.name!r} declares a path source without a path")

    normalized_path = str(_resolve_path_source(path, tap))
    if source.editable:
        return ["-e", normalized_path]
    return [normalized_path]


def _required(value: str | None, field_name: str, source_type: str) -> str:
    if value is None:
        raise ValueError(f"Plugin install source type {source_type!r} requires {field_name!r}")
    return value


def _resolve_path_source(path: str, tap: PluginTapConfig) -> Path:
    source_path = Path(path).expanduser()
    if source_path.is_absolute():
        return source_path

    tap_location = urlparse(tap.url)
    if tap_location.scheme in {"http", "https"}:
        raise ValueError("Relative path plugin sources require a local plugin tap")

    tap_catalog_path = Path(tap.url).expanduser()
    if tap_catalog_path.name == "plugins.json" and tap_catalog_path.parent.name == "catalog":
        return tap_catalog_path.parent.parent / source_path
    return tap_catalog_path.parent / source_path
