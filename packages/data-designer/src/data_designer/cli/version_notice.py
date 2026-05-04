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
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

from data_designer.config.utils.constants import DATA_DESIGNER_HOME

_PACKAGE_NAME = "data-designer"
_PYPI_JSON_URL = f"https://pypi.org/pypi/{_PACKAGE_NAME}/json"
_VERSION_CHECK_TIMEOUT_SECONDS = 0.75
_CACHE_TTL_SECONDS = 6 * 60 * 60
_CACHE_FILE_NAME = "version-check.json"
_DISABLE_VERSION_CHECK_ENV_VAR = "DATA_DESIGNER_DISABLE_VERSION_CHECK"
_INCLUDE_PRERELEASES_ENV_VAR = "DATA_DESIGNER_VERSION_CHECK_PRERELEASES"
_UV_TOOL_UPGRADE_COMMAND = "uv tool upgrade data-designer"
_PROJECT_UPGRADE_COMMAND = "uv add --upgrade data-designer"


@dataclass(frozen=True)
class UpdateNotice:
    latest_version: str
    upgrade_command: str


def get_update_notice(
    installed_version: str,
    *,
    cache_dir: Path = DATA_DESIGNER_HOME,
    environ: Mapping[str, str] | None = None,
    now: Callable[[], float] = time.time,
    python_prefix: str | None = None,
) -> UpdateNotice | None:
    env = os.environ if environ is None else environ
    if _env_flag_enabled(env, _DISABLE_VERSION_CHECK_ENV_VAR):
        return None

    try:
        installed = Version(installed_version)
    except InvalidVersion:
        return None

    include_prereleases = installed.is_prerelease or _env_flag_enabled(env, _INCLUDE_PRERELEASES_ENV_VAR)
    latest_version = _get_latest_version(
        include_prereleases=include_prereleases,
        cache_dir=cache_dir,
        now=now,
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
    prefix_parts = set(prefix.parts)
    if "uv" in prefix_parts and "tools" in prefix_parts:
        return _UV_TOOL_UPGRADE_COMMAND
    if env.get("UV_PROJECT_ENVIRONMENT") or env.get("VIRTUAL_ENV"):
        return _PROJECT_UPGRADE_COMMAND
    if prefix.name == ".venv":
        return _PROJECT_UPGRADE_COMMAND
    return _UV_TOOL_UPGRADE_COMMAND


def _get_latest_version(
    *,
    include_prereleases: bool,
    cache_dir: Path,
    now: Callable[[], float],
) -> str | None:
    cache_path = cache_dir / _CACHE_FILE_NAME
    cached_version = _read_cached_version(
        cache_path=cache_path,
        include_prereleases=include_prereleases,
        now=now,
    )
    if cached_version is not None:
        return cached_version

    try:
        latest_version = _fetch_latest_version(include_prereleases=include_prereleases)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    if latest_version is not None:
        _write_cached_version(
            cache_path=cache_path,
            latest_version=latest_version,
            include_prereleases=include_prereleases,
            checked_at=now(),
        )
    return latest_version


def _fetch_latest_version(*, include_prereleases: bool) -> str | None:
    request = Request(_PYPI_JSON_URL, headers={"Accept": "application/json", "User-Agent": "data-designer"})
    with urlopen(request, timeout=_VERSION_CHECK_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        return None
    return _latest_version_from_pypi_payload(payload, include_prereleases=include_prereleases)


def _latest_version_from_pypi_payload(payload: Mapping[str, Any], *, include_prereleases: bool) -> str | None:
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        return None

    candidates: list[Version] = []
    for version_text, release_files in releases.items():
        if not isinstance(version_text, str) or _is_yanked_release(release_files):
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


def _is_yanked_release(release_files: Any) -> bool:
    if not isinstance(release_files, list) or not release_files:
        return True
    return all(isinstance(release_file, dict) and release_file.get("yanked", False) for release_file in release_files)


def _read_cached_version(
    *,
    cache_path: Path,
    include_prereleases: bool,
    now: Callable[[], float],
) -> str | None:
    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cache_data, dict):
        return None

    if cache_data.get("include_prereleases") != include_prereleases:
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
    checked_at: float,
) -> None:
    cache_data = {
        "checked_at": checked_at,
        "include_prereleases": include_prereleases,
        "latest_version": latest_version,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
    except OSError:
        return


def _env_flag_enabled(env: Mapping[str, str], name: str) -> bool:
    value = env.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
