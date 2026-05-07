# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

from pytest import MonkeyPatch

from data_designer.cli import version_notice
from data_designer.cli.version_notice import (
    get_update_notice,
    latest_version_from_pypi_payload,
    select_upgrade_command,
)


def _current_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def test_get_update_notice_returns_notice_for_newer_stable_version(tmp_path: Path) -> None:
    mock_fetch = Mock(return_value="0.6.1")

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        python_prefix="/opt/python",
        fetch_latest_version=mock_fetch,
    )

    assert notice is not None
    assert notice.latest_version == "0.6.1"
    assert notice.upgrade_command == "pip install --upgrade data-designer"
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_get_update_notice_returns_none_for_current_version(tmp_path: Path) -> None:
    mock_fetch = Mock(return_value="0.6.0")

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is None
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_get_update_notice_fails_closed_when_check_fails(tmp_path: Path) -> None:
    mock_fetch = Mock(side_effect=OSError("network unavailable"))

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is None
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_get_update_notice_returns_none_for_invalid_installed_version(
    tmp_path: Path,
) -> None:
    mock_fetch = Mock(return_value="0.6.1")

    notice = get_update_notice(
        "not-a-version",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is None
    mock_fetch.assert_not_called()


def test_get_update_notice_returns_none_for_local_installed_version(
    tmp_path: Path,
) -> None:
    mock_fetch = Mock(return_value="0.6.1")

    notice = get_update_notice(
        "0.6.1.dev0+gabc1234",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is None
    mock_fetch.assert_not_called()


def test_get_update_notice_respects_opt_out(tmp_path: Path) -> None:
    mock_fetch = Mock(return_value="0.6.1")

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={"DATA_DESIGNER_DISABLE_VERSION_CHECK": "1"},
        now=lambda: 1_000.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is None
    mock_fetch.assert_not_called()


def test_get_update_notice_uses_fresh_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "version-check.json"
    cache_path.write_text(
        json.dumps(
            {
                "checked_at": 1_000.0,
                "include_prereleases": False,
                "latest_version": "0.6.1",
                "package_name": "data-designer",
                "python_version": _current_python_version(),
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    mock_fetch = Mock(return_value="0.6.2")

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_001.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is not None
    assert notice.latest_version == "0.6.1"
    mock_fetch.assert_not_called()


def test_get_update_notice_refetches_expired_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "version-check.json"
    cache_path.write_text(
        json.dumps(
            {
                "checked_at": 1_000.0,
                "include_prereleases": False,
                "latest_version": "0.6.1",
                "package_name": "data-designer",
                "python_version": _current_python_version(),
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    mock_fetch = Mock(return_value="0.6.2")

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0 + (7 * 60 * 60),
        fetch_latest_version=mock_fetch,
    )

    assert notice is not None
    assert notice.latest_version == "0.6.2"
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_get_update_notice_ignores_cache_with_old_schema(tmp_path: Path) -> None:
    cache_path = tmp_path / "version-check.json"
    cache_path.write_text(
        json.dumps(
            {
                "checked_at": 1_000.0,
                "include_prereleases": False,
                "latest_version": "0.6.1",
            }
        ),
        encoding="utf-8",
    )
    mock_fetch = Mock(return_value="0.6.2")

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_001.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is not None
    assert notice.latest_version == "0.6.2"
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_prerelease_versions_are_ignored_unless_requested() -> None:
    payload = {
        "releases": {
            "0.6.1": [{"yanked": False}],
            "0.6.2rc1": [{"yanked": False}],
        }
    }

    assert latest_version_from_pypi_payload(payload, include_prereleases=False) == "0.6.1"
    assert latest_version_from_pypi_payload(payload, include_prereleases=True) == "0.6.2rc1"


def test_latest_version_ignores_python_incompatible_release_files() -> None:
    payload = {
        "releases": {
            "0.6.1": [{"requires_python": ">=3.10", "yanked": False}],
            "0.6.2": [{"requires_python": ">=3.12", "yanked": False}],
        }
    }

    latest_version = latest_version_from_pypi_payload(
        payload,
        include_prereleases=False,
        python_version="3.11.0",
    )

    assert latest_version == "0.6.1"


def test_latest_version_ignores_invalid_requires_python() -> None:
    payload = {
        "releases": {
            "0.6.1": [{"requires_python": "not-a-specifier", "yanked": False}],
        }
    }

    assert (
        latest_version_from_pypi_payload(
            payload,
            include_prereleases=False,
            python_version="3.11.0",
        )
        is None
    )


def test_latest_version_ignores_yanked_and_malformed_release_files() -> None:
    payload = {
        "releases": {
            "0.6.1": [{"yanked": False}],
            "0.6.2": [{"yanked": True}, "not-a-file-record"],
            "0.6.3": [],
        }
    }

    assert latest_version_from_pypi_payload(payload, include_prereleases=False) == "0.6.1"


def test_latest_version_returns_none_for_malformed_pypi_payload() -> None:
    assert latest_version_from_pypi_payload({}, include_prereleases=False) is None
    assert latest_version_from_pypi_payload({"releases": []}, include_prereleases=False) is None


def test_get_update_notice_fetches_latest_version_from_pypi_payload(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    payload = json.dumps({"releases": {"0.6.1": [{"requires_python": ">=3.10", "yanked": False}]}}).encode()
    mock_urlopen = Mock(return_value=BytesIO(payload))
    monkeypatch.setattr(version_notice, "urlopen", mock_urlopen)

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        python_prefix="/opt/python",
    )

    assert notice is not None
    assert notice.latest_version == "0.6.1"
    assert notice.upgrade_command == "pip install --upgrade data-designer"
    assert mock_urlopen.call_args.kwargs["timeout"] == 0.75


def test_get_update_notice_fails_closed_when_urlopen_fails(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_urlopen = Mock(side_effect=URLError("blocked"))
    monkeypatch.setattr(version_notice, "urlopen", mock_urlopen)

    notice = get_update_notice("0.6.0", cache_dir=tmp_path, environ={}, now=lambda: 1_000.0)

    assert notice is None
    assert mock_urlopen.call_args.kwargs["timeout"] == 0.75


def test_installed_prerelease_opts_into_prerelease_checks(tmp_path: Path) -> None:
    mock_fetch = Mock(return_value="0.6.2rc2")

    notice = get_update_notice(
        "0.6.2rc1",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is not None
    assert notice.latest_version == "0.6.2rc2"
    mock_fetch.assert_called_once_with(include_prereleases=True)


def test_prerelease_environment_flag_opts_into_prerelease_checks(tmp_path: Path) -> None:
    mock_fetch = Mock(return_value="0.6.2rc1")

    notice = get_update_notice(
        "0.6.1",
        cache_dir=tmp_path,
        environ={"DATA_DESIGNER_VERSION_CHECK_PRERELEASES": "true"},
        now=lambda: 1_000.0,
        fetch_latest_version=mock_fetch,
    )

    assert notice is not None
    assert notice.latest_version == "0.6.2rc1"
    mock_fetch.assert_called_once_with(include_prereleases=True)


def test_select_upgrade_command_defaults_to_pip() -> None:
    command = select_upgrade_command(environ={}, python_prefix="/opt/python")

    assert command == "pip install --upgrade data-designer"


def test_select_upgrade_command_detects_uv_tool_environment() -> None:
    command = select_upgrade_command(
        environ={"VIRTUAL_ENV": "/repo/.venv"},
        python_prefix="/Users/user/.local/share/uv/tools/data-designer",
    )

    assert command == "uv tool upgrade data-designer"


def test_select_upgrade_command_treats_project_venv_under_uv_tools_as_project() -> None:
    command = select_upgrade_command(
        environ={},
        python_prefix="/Users/user/projects/uv/tools/my-project/.venv",
    )

    assert command == "pip install --upgrade data-designer"


def test_select_upgrade_command_ignores_nonmatching_uv_tool_paths() -> None:
    for python_prefix in (
        "/Users/user/uv/code/tools/data-designer",
        "/Users/user/.local/share/uv/tools/data-designer/bin",
    ):
        command = select_upgrade_command(
            environ={"VIRTUAL_ENV": "/repo/.venv"},
            python_prefix=python_prefix,
        )

        assert command == "pip install --upgrade data-designer"


def test_select_upgrade_command_detects_pipx_environment() -> None:
    command = select_upgrade_command(
        environ={},
        python_prefix="/Users/user/.local/pipx/venvs/data-designer",
    )

    assert command == "pipx upgrade data-designer"


def test_select_upgrade_command_ignores_nonmatching_pipx_paths() -> None:
    for python_prefix in (
        "/Users/user/.local/pipx/project/venvs/data-designer",
        "/Users/user/.local/pipx/venvs/data-designer/bin",
    ):
        command = select_upgrade_command(
            environ={"VIRTUAL_ENV": "/repo/.venv"},
            python_prefix=python_prefix,
        )

        assert command == "pip install --upgrade data-designer"


def test_select_upgrade_command_detects_project_environment() -> None:
    command = select_upgrade_command(
        environ={"UV_PROJECT_ENVIRONMENT": ".venv"},
        python_prefix="/repo/.venv",
    )

    assert command == "uv add --upgrade data-designer"


def test_select_upgrade_command_detects_plain_pip_virtualenv() -> None:
    command = select_upgrade_command(
        environ={"VIRTUAL_ENV": "/repo/.venv"},
        python_prefix="/repo/.venv",
    )

    assert command == "pip install --upgrade data-designer"
