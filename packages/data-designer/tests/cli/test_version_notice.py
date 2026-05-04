# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from pytest import MonkeyPatch

from data_designer.cli import version_notice
from data_designer.cli.version_notice import get_update_notice, select_upgrade_command


def test_get_update_notice_returns_notice_for_newer_stable_version(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    mock_fetch = Mock(return_value="0.6.1")
    monkeypatch.setattr(version_notice, "_fetch_latest_version", mock_fetch)

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={},
        now=lambda: 1_000.0,
        python_prefix="/opt/python",
    )

    assert notice is not None
    assert notice.latest_version == "0.6.1"
    assert notice.upgrade_command == "uv tool upgrade data-designer"
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_get_update_notice_returns_none_for_current_version(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    mock_fetch = Mock(return_value="0.6.0")
    monkeypatch.setattr(version_notice, "_fetch_latest_version", mock_fetch)

    notice = get_update_notice("0.6.0", cache_dir=tmp_path, environ={}, now=lambda: 1_000.0)

    assert notice is None
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_get_update_notice_fails_closed_when_check_fails(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    mock_fetch = Mock(side_effect=OSError("network unavailable"))
    monkeypatch.setattr(version_notice, "_fetch_latest_version", mock_fetch)

    notice = get_update_notice("0.6.0", cache_dir=tmp_path, environ={}, now=lambda: 1_000.0)

    assert notice is None
    mock_fetch.assert_called_once_with(include_prereleases=False)


def test_get_update_notice_respects_opt_out(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    mock_fetch = Mock(return_value="0.6.1")
    monkeypatch.setattr(version_notice, "_fetch_latest_version", mock_fetch)

    notice = get_update_notice(
        "0.6.0",
        cache_dir=tmp_path,
        environ={"DATA_DESIGNER_DISABLE_VERSION_CHECK": "1"},
        now=lambda: 1_000.0,
    )

    assert notice is None
    mock_fetch.assert_not_called()


def test_get_update_notice_uses_fresh_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
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
    monkeypatch.setattr(version_notice, "_fetch_latest_version", mock_fetch)

    notice = get_update_notice("0.6.0", cache_dir=tmp_path, environ={}, now=lambda: 1_001.0)

    assert notice is not None
    assert notice.latest_version == "0.6.1"
    mock_fetch.assert_not_called()


def test_prerelease_versions_are_ignored_unless_requested() -> None:
    payload = {
        "releases": {
            "0.6.1": [{"yanked": False}],
            "0.6.2rc1": [{"yanked": False}],
        }
    }

    assert version_notice._latest_version_from_pypi_payload(payload, include_prereleases=False) == "0.6.1"
    assert version_notice._latest_version_from_pypi_payload(payload, include_prereleases=True) == "0.6.2rc1"


def test_installed_prerelease_opts_into_prerelease_checks(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    mock_fetch = Mock(return_value="0.6.2rc2")
    monkeypatch.setattr(version_notice, "_fetch_latest_version", mock_fetch)

    notice = get_update_notice("0.6.2rc1", cache_dir=tmp_path, environ={}, now=lambda: 1_000.0)

    assert notice is not None
    assert notice.latest_version == "0.6.2rc2"
    mock_fetch.assert_called_once_with(include_prereleases=True)


def test_prerelease_environment_flag_opts_into_prerelease_checks(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    mock_fetch = Mock(return_value="0.6.2rc1")
    monkeypatch.setattr(version_notice, "_fetch_latest_version", mock_fetch)

    notice = get_update_notice(
        "0.6.1",
        cache_dir=tmp_path,
        environ={"DATA_DESIGNER_VERSION_CHECK_PRERELEASES": "true"},
        now=lambda: 1_000.0,
    )

    assert notice is not None
    assert notice.latest_version == "0.6.2rc1"
    mock_fetch.assert_called_once_with(include_prereleases=True)


def test_select_upgrade_command_defaults_to_uv_tool() -> None:
    command = select_upgrade_command(environ={}, python_prefix="/opt/python")

    assert command == "uv tool upgrade data-designer"


def test_select_upgrade_command_detects_uv_tool_environment() -> None:
    command = select_upgrade_command(
        environ={"VIRTUAL_ENV": "/repo/.venv"},
        python_prefix="/Users/user/.local/share/uv/tools/data-designer",
    )

    assert command == "uv tool upgrade data-designer"


def test_select_upgrade_command_detects_project_environment() -> None:
    command = select_upgrade_command(
        environ={"UV_PROJECT_ENVIRONMENT": ".venv"},
        python_prefix="/repo/.venv",
    )

    assert command == "uv add --upgrade data-designer"
