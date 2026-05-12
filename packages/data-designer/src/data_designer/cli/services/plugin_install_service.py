# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from data_designer.cli.plugin_catalog import (
    DATA_DESIGNER_DISTRIBUTION_NAME,
    PLUGIN_ENTRY_POINT_GROUP,
    PYPI_SIMPLE_INDEX_URL,
    InstallCommandTemporaryFile,
    InstallPlan,
    PluginCatalogConfig,
    PluginCatalogEntry,
    UninstallPlan,
)
from data_designer.plugins.plugin import Plugin

InstallRunner = Callable[[list[str], str | None], int]
PIP_EXTRA_INDEX_SOURCE_WARNING = (
    "pip --extra-index-url is not source-pinned; pip may choose a same-named package from another configured index. "
    "Use uv or a direct reference when strict source selection is required."
)
DATA_DESIGNER_DISTRIBUTION_NAMES = (
    DATA_DESIGNER_DISTRIBUTION_NAME,
    "data-designer-config",
    "data-designer-engine",
)
DATA_DESIGNER_PROJECT_NAMES = (*DATA_DESIGNER_DISTRIBUTION_NAMES, "data-designer-workspace")
PIP_DATA_DESIGNER_CONSTRAINT_FILE_NAME = "data-designer-constraint.txt"
DATA_DESIGNER_CONSTRAINT_PLACEHOLDER = "<temporary-data-designer-constraint-file>"
UV_PLUGIN_INSTALL_MIN_VERSION = Version("0.10.0")

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10.
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _InstallTarget:
    manager: str
    mode: str
    project_root: Path | None = None
    warning: str | None = None


class PluginInstallService:
    """Resolve, execute, and verify plugin package install and uninstall plans.

    When no working directory is provided, plan resolution uses the current
    process directory at build time so CLI calls follow the user's active shell.
    """

    def __init__(
        self,
        runner: InstallRunner | None = None,
        *,
        working_dir: Path | None = None,
        active_virtualenv: bool | None = None,
    ) -> None:
        self._runner = runner or _run_subprocess
        self._working_dir = working_dir
        self._active_virtualenv = active_virtualenv

    def build_install_plan(
        self,
        entry: PluginCatalogEntry,
        catalog: PluginCatalogConfig,
        *,
        manager: str = "auto",
        version_specifier: str | None = None,
    ) -> InstallPlan:
        """Build the exact package-manager command for one catalog entry."""
        target = _resolve_install_target(
            manager,
            working_dir=self._working_dir or Path.cwd(),
            active_virtualenv=self._active_virtualenv,
        )
        data_designer_versions = _installed_data_designer_distribution_versions()
        protection_args, data_designer_protection, command_stdin, temporary_file = _data_designer_protection_args(
            target.mode,
            data_designer_versions,
        )
        install_args, install_requirement, source_description, source_warning = _install_args_for_entry(
            entry,
            target,
            version_specifier=version_specifier,
        )
        command = _base_command(target) + protection_args + install_args
        return InstallPlan(
            package_name=entry.package.name,
            source_description=source_description,
            command=command,
            manager=target.manager,
            catalog_alias=catalog.alias,
            requirement=install_requirement,
            source_warning=_combine_warnings(target.warning, source_warning),
            data_designer_protection=data_designer_protection,
            data_designer_version=data_designer_versions[DATA_DESIGNER_DISTRIBUTION_NAME],
            command_stdin=command_stdin,
            temporary_file=temporary_file,
            install_mode=target.mode,
            project_root=str(target.project_root) if target.project_root is not None else None,
        )

    def build_uninstall_plan(
        self,
        entry: PluginCatalogEntry,
        catalog: PluginCatalogConfig,
        *,
        manager: str = "auto",
    ) -> UninstallPlan:
        """Build the exact package-manager command to uninstall one catalog package."""
        target = _resolve_install_target(
            manager,
            working_dir=self._working_dir or Path.cwd(),
            active_virtualenv=self._active_virtualenv,
        )
        commands = _uninstall_commands(target, entry.package.name)
        return UninstallPlan(
            package_name=entry.package.name,
            command=commands[0],
            manager=target.manager,
            catalog_alias=catalog.alias,
            source_warning=target.warning,
            commands=commands,
            uninstall_mode=target.mode,
            project_root=str(target.project_root) if target.project_root is not None else None,
        )

    def install(self, plan: InstallPlan) -> None:
        """Run an installation plan.

        Raises:
            RuntimeError: If the package manager exits unsuccessfully.
        """
        with _materialized_install_command(plan) as (command, command_stdin):
            return_code = self._runner(command, command_stdin)
        if return_code != 0:
            raise RuntimeError(f"Plugin package installer exited with status {return_code}")

    def uninstall(self, plan: UninstallPlan) -> None:
        """Run an uninstall plan.

        Raises:
            RuntimeError: If the package manager exits unsuccessfully.
        """
        for command in plan.commands or [plan.command]:
            return_code = self._runner(command, None)
            if return_code != 0:
                raise RuntimeError(f"Plugin package uninstaller exited with status {return_code}")

    def verify_entry_point(self, entry: PluginCatalogEntry) -> bool:
        """Verify the runtime plugin's declared entry point is installed and loadable."""
        return self.verify_entry_points([entry])

    def verify_entry_points(self, entries: list[PluginCatalogEntry]) -> bool:
        """Verify every declared runtime entry point for an installed catalog package can load."""
        if not entries:
            return False

        importlib.invalidate_caches()
        installed_entry_points = list(importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP))
        return all(_matching_entry_point_loads_plugin(entry, installed_entry_points) for entry in entries)

    def verify_entry_points_removed(self, entries: list[PluginCatalogEntry]) -> bool:
        """Verify every declared runtime entry point for a catalog package is no longer installed."""
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


def _run_subprocess(command: list[str], stdin_text: str | None) -> int:
    if stdin_text is None:
        result = subprocess.run(command, check=False, stdin=subprocess.DEVNULL)
    else:
        result = subprocess.run(command, check=False, input=stdin_text, text=True)
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


def _matching_entry_point_loads_plugin(
    entry: PluginCatalogEntry,
    installed_entry_points: list[importlib.metadata.EntryPoint],
) -> bool:
    for installed_entry_point in installed_entry_points:
        if not _installed_entry_point_matches(installed_entry_point, entry):
            continue
        if _entry_point_loads_plugin(installed_entry_point):
            return True
    return False


def _entry_point_loads_plugin(installed_entry_point: importlib.metadata.EntryPoint) -> bool:
    try:
        plugin = installed_entry_point.load()
    except (Exception, SystemExit):
        return False
    return isinstance(plugin, Plugin)


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


def _resolve_install_target(
    manager: str,
    *,
    working_dir: Path,
    active_virtualenv: bool | None,
) -> _InstallTarget:
    if manager not in {"auto", "uv", "pip"}:
        raise ValueError(f"Unsupported plugin installer {manager!r}. Expected 'auto', 'uv', or 'pip'.")

    uv_path = shutil.which("uv") if manager in {"auto", "uv"} else None
    if manager == "auto":
        if uv_path is None:
            return _pip_install_target(unavailable_context="Auto mode could not find uv on PATH and needs pip fallback")
        uv_warning = _uv_plugin_install_error(uv_path)
        if uv_warning is not None:
            pip_error = _pip_install_error()
            if pip_error is not None:
                raise ValueError(f"{uv_warning}\n\nAuto mode needs pip fallback, but {pip_error}")
            return _InstallTarget(
                manager="pip",
                mode="pip-environment",
                warning=f"{uv_warning}\n\n{_uv_recovery_message(auto_fallback=True)}",
            )
        return _uv_install_target(working_dir, active_virtualenv)

    if manager == "uv":
        if uv_path is None:
            raise ValueError(
                "uv was requested for plugin package installation, but it is not available on PATH. "
                "Install uv or pass --manager pip to use pip."
            )
        uv_error = _uv_plugin_install_error(uv_path)
        if uv_error is not None:
            raise ValueError(f"{uv_error}\n\n{_uv_recovery_message(auto_fallback=False)}")
        return _uv_install_target(working_dir, active_virtualenv)

    return _pip_install_target(unavailable_context="pip was requested for plugin package installation")


def _pip_install_target(*, unavailable_context: str) -> _InstallTarget:
    pip_error = _pip_install_error()
    if pip_error is not None:
        raise ValueError(f"{unavailable_context}, but {pip_error}")
    return _InstallTarget(manager="pip", mode="pip-environment")


def _uv_install_target(
    working_dir: Path,
    active_virtualenv: bool | None,
) -> _InstallTarget:
    project_root = _project_root_for_uv_add(working_dir, active_virtualenv)
    if project_root is not None:
        return _InstallTarget(manager="uv", mode="uv-project", project_root=project_root)

    return _InstallTarget(manager="uv", mode="uv-environment")


def _base_command(target: _InstallTarget) -> list[str]:
    if target.mode == "uv-project":
        if target.project_root is None:
            raise ValueError("uv project install target requires a project root")
        return ["uv", "add", "--project", str(target.project_root), "--active", "--no-install-project"]
    if target.mode == "uv-environment":
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _uninstall_commands(target: _InstallTarget, package_name: str) -> list[list[str]]:
    if target.mode == "uv-project":
        if target.project_root is None:
            raise ValueError("uv project uninstall target requires a project root")
        return [
            ["uv", "remove", "--project", str(target.project_root), "--no-sync", package_name],
            ["uv", "pip", "uninstall", "--python", sys.executable, package_name],
        ]
    return [_base_uninstall_command(target) + [package_name]]


def _base_uninstall_command(target: _InstallTarget) -> list[str]:
    if target.manager == "uv":
        return ["uv", "pip", "uninstall", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "uninstall", "--yes"]


def _project_root_for_uv_add(working_dir: Path, active_virtualenv: bool | None) -> Path | None:
    if not _has_active_virtualenv(active_virtualenv):
        return None

    project_root = _find_nearest_pyproject_root(working_dir)
    if project_root is None or _is_data_designer_source_project(project_root):
        return None
    return project_root


def _has_active_virtualenv(active_virtualenv: bool | None) -> bool:
    if active_virtualenv is not None:
        return active_virtualenv
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.getenv("VIRTUAL_ENV"))


def _pip_install_error() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"pip is not available for {sys.executable!r}: {e}."

    if result.returncode == 0:
        return None

    output = (result.stderr or result.stdout).strip()
    details = f": {output}" if output else ""
    return f"`{sys.executable} -m pip --version` exited with status {result.returncode}{details}."


def _uv_plugin_install_error(uv_path: str) -> str | None:
    try:
        result = subprocess.run(
            [uv_path, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"Unable to verify uv at {uv_path!r}: {e}."

    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        details = f": {output}" if output else ""
        return f"Unable to verify uv at {uv_path!r}; uv --version exited with status {result.returncode}{details}."

    uv_version = _parse_uv_version(output)
    if uv_version is None:
        return (
            f"Unable to parse uv version from {output!r}; Data Designer plugin package commands with uv require "
            f"uv >= {UV_PLUGIN_INSTALL_MIN_VERSION}."
        )
    if uv_version < UV_PLUGIN_INSTALL_MIN_VERSION:
        return (
            f"Found uv {uv_version}, but Data Designer plugin package commands with uv require "
            f"uv >= {UV_PLUGIN_INSTALL_MIN_VERSION}."
        )
    return None


def _uv_recovery_message(*, auto_fallback: bool) -> str:
    if auto_fallback:
        return "Auto mode will use pip for this plan. Upgrade uv with `uv self update` to use uv."
    return "Upgrade uv with `uv self update` or pass --manager pip to use pip."


def _parse_uv_version(output: str) -> Version | None:
    for token in output.split():
        try:
            return Version(token)
        except InvalidVersion:
            continue
    return None


def _find_nearest_pyproject_root(working_dir: Path) -> Path | None:
    resolved_working_dir = working_dir.resolve()
    for candidate in (resolved_working_dir, *resolved_working_dir.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _is_data_designer_source_project(project_root: Path) -> bool:
    pyproject_data = _load_pyproject_data(project_root / "pyproject.toml")
    project = pyproject_data.get("project", {})
    if isinstance(project, dict):
        project_name = project.get("name")
        if isinstance(project_name, str) and canonicalize_name(project_name) in DATA_DESIGNER_PROJECT_NAMES:
            return True

    try:
        relative_source_file = Path(__file__).resolve().relative_to(project_root.resolve())
    except (OSError, ValueError):
        return False
    source_parts = relative_source_file.parts
    return source_parts[:3] == ("packages", "data-designer", "src") or source_parts[:2] == ("src", "data_designer")


def _load_pyproject_data(pyproject_path: Path) -> dict[str, Any]:
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    return _load_pyproject_markers_without_tomllib(text)


def _load_pyproject_markers_without_tomllib(text: str) -> dict[str, Any]:
    # Python 3.10 only needs a deliberately lossy fallback: detect simple
    # [project] name markers from this repo's pyprojects, not parse TOML.
    project: dict[str, Any] = {}
    section = ""

    for raw_line in text.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        if "=" not in line:
            continue

        key, raw_value = (part.strip() for part in line.split("=", maxsplit=1))
        if section == "project" and key == "name":
            project["name"] = _parse_simple_toml_value(raw_value)

    data: dict[str, Any] = {}
    if project:
        data["project"] = project
    return data


def _parse_simple_toml_value(raw_value: str) -> str | None:
    value = raw_value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return None


def _installed_data_designer_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution_name in DATA_DESIGNER_DISTRIBUTION_NAMES:
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError as e:
            raise ValueError(
                f"Unable to resolve installed {distribution_name!r} version; "
                "plugin package installs require the Data Designer package family to be installed first."
            ) from e

        try:
            Version(version)
        except InvalidVersion as e:
            raise ValueError(
                f"Installed {distribution_name!r} version {version!r} is not a valid package version; "
                "cannot protect the current Data Designer installation during plugin package install."
            ) from e
        versions[distribution_name] = version
    return versions


def _data_designer_protection_args(
    mode: str,
    versions: dict[str, str],
) -> tuple[list[str], str, str | None, InstallCommandTemporaryFile | None]:
    data_designer_version = versions[DATA_DESIGNER_DISTRIBUTION_NAME]
    if mode == "uv-environment":
        return (
            ["--constraint", "-"],
            f"using installed {DATA_DESIGNER_DISTRIBUTION_NAME} {data_designer_version}; "
            "uv will keep Data Designer packages pinned",
            _data_designer_constraint_text(versions),
            None,
        )

    if mode == "uv-project":
        return (
            [
                *[
                    item
                    for distribution_name in DATA_DESIGNER_DISTRIBUTION_NAMES
                    for item in ("--no-install-package", distribution_name)
                ],
            ],
            f"using installed {DATA_DESIGNER_DISTRIBUTION_NAME} {data_designer_version}; "
            "uv will not install Data Designer packages",
            None,
            None,
        )

    return (
        ["--constraint", DATA_DESIGNER_CONSTRAINT_PLACEHOLDER],
        f"pinned installed Data Designer packages; {DATA_DESIGNER_DISTRIBUTION_NAME} {data_designer_version}",
        None,
        _data_designer_constraint_file(versions),
    )


def _data_designer_constraint_file(versions: dict[str, str]) -> InstallCommandTemporaryFile:
    return InstallCommandTemporaryFile(
        placeholder=DATA_DESIGNER_CONSTRAINT_PLACEHOLDER,
        filename=PIP_DATA_DESIGNER_CONSTRAINT_FILE_NAME,
        content=f"# Data Designer is provided by the active CLI environment.\n{_data_designer_constraint_text(versions)}",
    )


def _data_designer_constraint_text(versions: dict[str, str]) -> str:
    constraints = "\n".join(
        f"{distribution_name}=={versions[distribution_name]}" for distribution_name in DATA_DESIGNER_DISTRIBUTION_NAMES
    )
    return f"{constraints}\n"


@contextmanager
def _materialized_install_command(plan: InstallPlan) -> Iterator[tuple[list[str], str | None]]:
    temporary_file = plan.temporary_file
    if temporary_file is None:
        yield plan.command, plan.command_stdin
        return

    with tempfile.TemporaryDirectory(prefix="data-designer-plugin-install-") as temp_dir:
        temporary_path = Path(temp_dir) / temporary_file.filename
        temporary_path.write_text(temporary_file.content, encoding="utf-8")
        command = [str(temporary_path) if part == temporary_file.placeholder else part for part in plan.command]
        yield command, plan.command_stdin


def _install_args_for_entry(
    entry: PluginCatalogEntry,
    target: _InstallTarget,
    *,
    version_specifier: str | None,
) -> tuple[list[str], str, str, str | None]:
    requirement = _install_requirement_for_entry(entry, version_specifier=version_specifier)
    index_url = entry.install.index_url
    if target.mode == "uv-project":
        args = ["--raw"] if index_url is None and _requirement_is_direct_reference(requirement) else []
        if index_url is not None:
            args.extend(["--index", index_url])
        args.append(requirement)
        return args, requirement, _source_description(requirement, index_url), None

    if index_url is None:
        return [requirement], requirement, requirement, None

    if target.manager == "uv":
        return (
            ["--default-index", PYPI_SIMPLE_INDEX_URL, "--index", index_url, requirement],
            requirement,
            f"{requirement} via {index_url}",
            None,
        )
    return (
        ["--extra-index-url", index_url, requirement],
        requirement,
        f"{requirement} via {index_url}",
        PIP_EXTRA_INDEX_SOURCE_WARNING,
    )


def _install_requirement_for_entry(entry: PluginCatalogEntry, *, version_specifier: str | None) -> str:
    requirement = entry.install.requirement
    if version_specifier is None:
        return requirement

    try:
        parsed_requirement = Requirement(requirement)
    except InvalidRequirement as e:  # pragma: no cover - catalog validation catches this earlier.
        raise ValueError(f"Catalog install requirement {requirement!r} is invalid: {e}") from e

    if parsed_requirement.url is not None:
        raise ValueError(
            f"Cannot install a specific version for plugin package {entry.package.name!r} because the catalog "
            "install requirement is a direct reference."
        )

    extras = f"[{','.join(sorted(parsed_requirement.extras))}]" if parsed_requirement.extras else ""
    marker = f"; {parsed_requirement.marker}" if parsed_requirement.marker is not None else ""
    return f"{entry.package.name}{extras}{version_specifier}{marker}"


def _source_description(requirement: str, index_url: str | None) -> str:
    if index_url is None:
        return requirement
    return f"{requirement} via {index_url}"


def _combine_warnings(*warnings: str | None) -> str | None:
    active_warnings = [warning for warning in warnings if warning]
    if not active_warnings:
        return None
    return "\n".join(active_warnings)


def _requirement_is_direct_reference(requirement: str) -> bool:
    try:
        return Requirement(requirement).url is not None
    except InvalidRequirement:
        return " @ " in requirement
