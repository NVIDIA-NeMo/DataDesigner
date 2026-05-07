# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from data_designer.config.utils.constants import DATA_DESIGNER_HOME, DATA_DESIGNER_PACKAGE_NAME

_PYPI_JSON_URL = f"https://pypi.org/pypi/{DATA_DESIGNER_PACKAGE_NAME}/json"
_VERSION_CHECK_TIMEOUT_SECONDS = 0.75
_CACHE_TTL_SECONDS = 6 * 60 * 60
_CACHE_SCHEMA_VERSION = 1
_CACHE_FILE_NAME = "version-check.json"
_DISABLE_VERSION_CHECK_ENV_VAR = "DATA_DESIGNER_DISABLE_VERSION_CHECK"
_INCLUDE_PRERELEASES_ENV_VAR = "DATA_DESIGNER_VERSION_CHECK_PRERELEASES"
_UV_TOOL_UPGRADE_COMMAND = f"uv tool upgrade {DATA_DESIGNER_PACKAGE_NAME}"
_PROJECT_UPGRADE_COMMAND = f"uv add --upgrade {DATA_DESIGNER_PACKAGE_NAME}"
_PIPX_UPGRADE_COMMAND = f"pipx upgrade {DATA_DESIGNER_PACKAGE_NAME}"
_PIP_UPGRADE_COMMAND = f"pip install --upgrade {DATA_DESIGNER_PACKAGE_NAME}"


@dataclass(frozen=True)
class UpdateNotice:
    latest_version: str
    upgrade_command: str


class LatestVersionFetcher(Protocol):
    def __call__(self, *, include_prereleases: bool) -> str | None: ...


def get_update_notice(
    installed_version: str,
    *,
    cache_dir: Path = DATA_DESIGNER_HOME,
    environ: Mapping[str, str] | None = None,
    now: Callable[[], float] = time.time,
    python_prefix: str | None = None,
    fetch_latest_version: LatestVersionFetcher | None = None,
) -> UpdateNotice | None:
    env = os.environ if environ is None else environ
    if _env_flag_enabled(env, _DISABLE_VERSION_CHECK_ENV_VAR):
        return None

    try:
        installed = Version(installed_version)
    except InvalidVersion:
        return None
    if installed.local is not None:
        return None

    include_prereleases = installed.is_prerelease or _env_flag_enabled(env, _INCLUDE_PRERELEASES_ENV_VAR)
    latest_version = _get_latest_version(
        include_prereleases=include_prereleases,
        cache_dir=cache_dir,
        now=now,
        fetch_latest_version=_fetch_latest_version if fetch_latest_version is None else fetch_latest_version,
    )
    if latest_version is None:
        return None

    try:
        latest = Version(latest_version)
    except InvalidVersion:
        return None

    if latest <= installed:
        return None

    return UpdateNotice(
        latest_version=latest.public,
        upgrade_command=select_upgrade_command(environ=env, python_prefix=python_prefix),
    )


def select_upgrade_command(
    *,
    environ: Mapping[str, str] | None = None,
    python_prefix: str | None = None,
) -> str:
    env = os.environ if environ is None else environ
    prefix = Path(sys.prefix if python_prefix is None else python_prefix)
    prefix_parts = prefix.parts
    if _path_ends_with_segments(prefix_parts, "pipx", "venvs"):
        return _PIPX_UPGRADE_COMMAND
    if _path_ends_with_segments(prefix_parts, "uv", "tools"):
        return _UV_TOOL_UPGRADE_COMMAND
    if env.get("UV_PROJECT_ENVIRONMENT"):
        return _PROJECT_UPGRADE_COMMAND
    if env.get("VIRTUAL_ENV"):
        return _PIP_UPGRADE_COMMAND
    return _PIP_UPGRADE_COMMAND


def _path_ends_with_segments(parts: tuple[str, ...], parent: str, child: str) -> bool:
    """Return True when parts ends in .../{parent}/{child}/<one segment>."""
    return len(parts) >= 3 and parts[-3] == parent and parts[-2] == child


def _get_latest_version(
    *,
    include_prereleases: bool,
    cache_dir: Path,
    now: Callable[[], float],
    fetch_latest_version: LatestVersionFetcher,
) -> str | None:
    cache_path = cache_dir / _CACHE_FILE_NAME
    python_version = _current_python_version()
    cached_version = _read_cached_version(
        cache_path=cache_path,
        include_prereleases=include_prereleases,
        python_version=python_version,
        now=now,
    )
    if cached_version is not None:
        return cached_version

    try:
        latest_version = fetch_latest_version(include_prereleases=include_prereleases)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    if latest_version is not None:
        _write_cached_version(
            cache_path=cache_path,
            latest_version=latest_version,
            include_prereleases=include_prereleases,
            python_version=python_version,
            checked_at=now(),
        )
    return latest_version


def _fetch_latest_version(*, include_prereleases: bool) -> str | None:
    request = Request(_PYPI_JSON_URL, headers={"Accept": "application/json", "User-Agent": "data-designer"})
    with urlopen(request, timeout=_VERSION_CHECK_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        return None
    return latest_version_from_pypi_payload(payload, include_prereleases=include_prereleases)


def latest_version_from_pypi_payload(
    payload: Mapping[str, Any],
    *,
    include_prereleases: bool,
    python_version: str | None = None,
) -> str | None:
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        return None

    python_version = _current_python_version() if python_version is None else python_version
    candidates: list[Version] = []
    for version_text, release_files in releases.items():
        if not isinstance(version_text, str) or not _has_installable_release_file(
            release_files,
            python_version=python_version,
        ):
            continue
        try:
            version = Version(version_text)
        except InvalidVersion:
            continue
        if version.is_prerelease and not include_prereleases:
            continue
        candidates.append(version)

    if not candidates:
        return None

    return max(candidates).public


def _has_installable_release_file(release_files: Any, *, python_version: str) -> bool:
    if not isinstance(release_files, list):
        return False
    return any(
        isinstance(release_file, dict)
        and not release_file.get("yanked", False)
        and _is_python_compatible(release_file.get("requires_python"), python_version=python_version)
        for release_file in release_files
    )


def _is_python_compatible(requires_python: Any, *, python_version: str) -> bool:
    if requires_python in (None, ""):
        return True
    if not isinstance(requires_python, str):
        return False
    try:
        return SpecifierSet(requires_python).contains(python_version, prereleases=True)
    except InvalidSpecifier:
        return False


def _current_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _read_cached_version(
    *,
    cache_path: Path,
    include_prereleases: bool,
    python_version: str,
    now: Callable[[], float],
) -> str | None:
    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cache_data, dict):
        return None

    if cache_data.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return None
    if cache_data.get("package_name") != DATA_DESIGNER_PACKAGE_NAME:
        return None
    if cache_data.get("include_prereleases") != include_prereleases:
        return None
    if cache_data.get("python_version") != python_version:
        return None

    checked_at = cache_data.get("checked_at")
    latest_version = cache_data.get("latest_version")
    if not isinstance(checked_at, (int, float)) or not isinstance(latest_version, str):
        return None
    if now() - float(checked_at) > _CACHE_TTL_SECONDS:
        return None
    return latest_version


def _write_cached_version(
    *,
    cache_path: Path,
    latest_version: str,
    include_prereleases: bool,
    python_version: str,
    checked_at: float,
) -> None:
    cache_data = {
        "checked_at": checked_at,
        "include_prereleases": include_prereleases,
        "latest_version": latest_version,
        "package_name": DATA_DESIGNER_PACKAGE_NAME,
        "python_version": python_version,
        "schema_version": _CACHE_SCHEMA_VERSION,
    }
    temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(cache_data), encoding="utf-8")
        temp_path.replace(cache_path)
    except OSError:
        try:
            temp_path.unlink()
        except OSError:
            pass
        return


def _env_flag_enabled(env: Mapping[str, str], name: str) -> bool:
    value = env.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
