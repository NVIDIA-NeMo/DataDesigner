# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from data_designer.cli.plugin_catalog import PluginCatalogConfig, PluginCatalogEntry
from data_designer.cli.services.plugin_install_service import PIP_EXTRA_INDEX_SOURCE_WARNING, PluginInstallService
from data_designer.plugins.plugin import Plugin, PluginType

DATA_DESIGNER_VERSION = "0.5.10"
PIP_VERSION_PROBE = [sys.executable, "-m", "pip", "--version"]


@pytest.fixture(autouse=True)
def mock_data_designer_version() -> Iterator[None]:
    with (
        patch(
            "data_designer.cli.services.plugin_install_service.importlib.metadata.version",
            return_value=DATA_DESIGNER_VERSION,
        ),
        patch(
            "data_designer.cli.services.plugin_install_service.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="uv 0.10.0\n", stderr=""),
        ),
    ):
        yield


def test_build_pip_install_plan_uses_requirement_and_extra_index() -> None:
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="pip")

    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--constraint",
        "<temporary-data-designer-constraint-file>",
        "--extra-index-url",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template",
    ]
    assert plan.requirement == "data-designer-template"
    assert plan.temporary_file is not None
    assert plan.temporary_file.filename == "data-designer-constraint.txt"
    assert plan.temporary_file.content == (
        "# Data Designer is provided by the active CLI environment.\n"
        f"data-designer=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-config=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-engine=={DATA_DESIGNER_VERSION}\n"
    )
    assert plan.command_stdin is None
    assert plan.data_designer_version == DATA_DESIGNER_VERSION
    assert plan.source_warning == PIP_EXTRA_INDEX_SOURCE_WARNING


def test_build_pip_install_plan_applies_version_specifier() -> None:
    entry = _entry(
        package_name="data-designer-github",
        install={
            "requirement": "data-designer-github",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="pip", version_specifier="==0.1.0")

    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--constraint",
        "<temporary-data-designer-constraint-file>",
        "--extra-index-url",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-github==0.1.0",
    ]
    assert plan.requirement == "data-designer-github==0.1.0"


def test_build_pip_install_plan_preserves_catalog_specifier_for_versioned_install() -> None:
    entry = _entry(
        package_name="data-designer-github",
        install={
            "requirement": "data-designer-github>=0.2",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="pip", version_specifier="==0.1.0")

    assert plan.command[-1] == "data-designer-github>=0.2,==0.1.0"
    assert plan.requirement == "data-designer-github>=0.2,==0.1.0"


def test_build_pip_install_plan_preserves_catalog_extras_and_marker_for_versioned_install() -> None:
    entry = _entry(
        package_name="data-designer-github",
        install={
            "requirement": 'data-designer-github[cli]>=0.1; python_version >= "3.10"',
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="pip", version_specifier="==0.2.0")

    assert plan.command[-1] == 'data-designer-github[cli]>=0.1,==0.2.0; python_version >= "3.10"'
    assert plan.requirement == 'data-designer-github[cli]>=0.1,==0.2.0; python_version >= "3.10"'


def test_build_direct_reference_install_plan_uses_requirement_verbatim() -> None:
    requirement = (
        "data-designer-template @ "
        "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@data-designer-template/v0.1.0"
    )
    entry = _entry(package_name="data-designer-template", install={"requirement": requirement})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="pip")

    assert plan.command[-1] == requirement
    assert "--extra-index-url" not in plan.command
    assert plan.source_warning is None


def test_build_install_plan_rejects_version_for_direct_reference() -> None:
    requirement = (
        "data-designer-template @ "
        "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@data-designer-template/v0.1.0"
    )
    entry = _entry(package_name="data-designer-template", install={"requirement": requirement})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with pytest.raises(ValueError, match="direct reference"):
        service.build_install_plan(entry, catalog, manager="pip", version_specifier="==0.1.1")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_chooses_uv_when_available(mock_which: Mock) -> None:
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.manager == "uv"
    assert plan.install_mode == "uv-environment"
    assert plan.command == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--constraint",
        "-",
        "--default-index",
        "https://pypi.org/simple/",
        "--index",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template",
    ]
    assert plan.command_stdin == (
        f"data-designer=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-config=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-engine=={DATA_DESIGNER_VERSION}\n"
    )
    assert plan.temporary_file is None
    assert plan.data_designer_version == DATA_DESIGNER_VERSION
    assert plan.source_warning is None
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_uses_uv_add_for_active_project(mock_which: Mock, tmp_path: Path) -> None:
    working_dir = _write_project(tmp_path) / "src"
    working_dir.mkdir()
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService(working_dir=working_dir, active_virtualenv=True)

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.manager == "uv"
    assert plan.install_mode == "uv-project"
    assert plan.project_root == str(tmp_path)
    assert plan.command == [
        "uv",
        "add",
        "--project",
        str(tmp_path),
        "--active",
        "--no-install-project",
        "--no-install-package",
        "data-designer",
        "--no-install-package",
        "data-designer-config",
        "--no-install-package",
        "data-designer-engine",
        "--index",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template",
    ]
    assert plan.command_stdin is None
    assert plan.temporary_file is None
    assert plan.source_warning is None
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_does_not_use_uv_add_without_active_virtualenv(
    mock_which: Mock,
    tmp_path: Path,
) -> None:
    _write_project(tmp_path)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService(working_dir=tmp_path, active_virtualenv=False)

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.install_mode == "uv-environment"
    assert plan.command[:6] == ["uv", "pip", "install", "--python", sys.executable, "--constraint"]
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_does_not_use_uv_add_for_data_designer_workspace(
    mock_which: Mock,
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "data-designer-workspace"\n[tool.uv]\npackage = false\n',
        encoding="utf-8",
    )
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService(working_dir=tmp_path, active_virtualenv=True)

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.install_mode == "uv-environment"
    assert plan.project_root is None
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_does_not_use_uv_add_for_active_virtualenv_without_pyproject(
    mock_which: Mock,
    tmp_path: Path,
) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService(working_dir=tmp_path, active_virtualenv=True)

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.manager == "uv"
    assert plan.install_mode == "uv-environment"
    assert plan.project_root is None
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_uses_uv_add_for_non_package_user_project(
    mock_which: Mock,
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "experiment-workspace"\n[tool.uv]\npackage = false\n',
        encoding="utf-8",
    )
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService(working_dir=tmp_path, active_virtualenv=True)

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.install_mode == "uv-project"
    assert plan.project_root == str(tmp_path)
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value=None)
def test_build_auto_install_plan_chooses_pip_when_uv_is_unavailable(mock_which: Mock) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.manager == "pip"
    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--constraint",
        "<temporary-data-designer-constraint-file>",
        "data-designer-template",
    ]
    assert plan.temporary_file is not None
    mock_which.assert_called_once_with("uv")


@patch(
    "data_designer.cli.services.plugin_install_service.subprocess.run",
    return_value=SimpleNamespace(returncode=0, stdout="uv 0.7.22\n", stderr=""),
)
@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_uses_pip_with_warning_when_uv_is_too_old(
    mock_which: Mock,
    mock_run: Mock,
) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="auto")

    assert plan.manager == "pip"
    assert plan.install_mode == "pip-environment"
    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--constraint",
        "<temporary-data-designer-constraint-file>",
        "data-designer-template",
    ]
    message = plan.source_warning or ""
    assert "Found uv 0.7.22" in message
    assert "uv >= 0.10.0" in message
    assert "uv self update" in message
    assert "Auto mode will use pip for this plan" in message
    assert "--manager pip" not in message
    mock_which.assert_called_once_with("uv")
    assert mock_run.call_count == 2
    assert [call.args[0] for call in mock_run.call_args_list] == [
        ["/usr/bin/uv", "--version"],
        PIP_VERSION_PROBE,
    ]


@patch(
    "data_designer.cli.services.plugin_install_service.subprocess.run",
    return_value=SimpleNamespace(returncode=1, stdout="", stderr="No module named pip"),
)
@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value=None)
def test_build_auto_install_plan_raises_when_uv_and_pip_are_unavailable(
    mock_which: Mock,
    mock_run: Mock,
) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with pytest.raises(ValueError) as exc_info:
        service.build_install_plan(entry, catalog, manager="auto")

    message = str(exc_info.value)
    assert "could not find uv on PATH" in message
    assert "pip" in message
    assert "No module named pip" in message
    mock_which.assert_called_once_with("uv")
    mock_run.assert_called_once()


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_install_plan_raises_when_old_uv_fallback_lacks_pip(mock_which: Mock) -> None:
    def run_package_manager_probe(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command == ["/usr/bin/uv", "--version"]:
            return SimpleNamespace(returncode=0, stdout="uv 0.7.22\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="No module named pip")

    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with (
        patch(
            "data_designer.cli.services.plugin_install_service.subprocess.run",
            side_effect=run_package_manager_probe,
        ) as mock_run,
        pytest.raises(ValueError) as exc_info,
    ):
        service.build_install_plan(entry, catalog, manager="auto")

    message = str(exc_info.value)
    assert "Found uv 0.7.22" in message
    assert "needs pip fallback" in message
    assert "No module named pip" in message
    mock_which.assert_called_once_with("uv")
    assert mock_run.call_count == 2


def test_build_pip_uninstall_plan_uses_package_name_not_install_requirement() -> None:
    requirement = (
        "data-designer-template @ "
        "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@data-designer-template/v0.1.0"
    )
    entry = _entry(package_name="data-designer-template", install={"requirement": requirement})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_uninstall_plan(entry, catalog, manager="pip")

    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "--yes",
        "data-designer-template",
    ]
    assert plan.package_name == "data-designer-template"
    assert plan.manager == "pip"


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_uninstall_plan_chooses_uv_when_available(mock_which: Mock) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_uninstall_plan(entry, catalog, manager="auto")

    assert plan.command == [
        "uv",
        "pip",
        "uninstall",
        "--python",
        sys.executable,
        "data-designer-template",
    ]
    assert plan.manager == "uv"
    assert plan.uninstall_mode == "uv-environment"
    mock_which.assert_called_once_with("uv")


@patch(
    "data_designer.cli.services.plugin_install_service.subprocess.run",
    return_value=SimpleNamespace(returncode=0, stdout="uv 0.7.22\n", stderr=""),
)
@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_uninstall_plan_uses_pip_with_warning_when_uv_is_too_old(
    mock_which: Mock,
    mock_run: Mock,
) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_uninstall_plan(entry, catalog, manager="auto")

    assert plan.manager == "pip"
    assert plan.uninstall_mode == "pip-environment"
    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "--yes",
        "data-designer-template",
    ]
    message = plan.source_warning or ""
    assert "Found uv 0.7.22" in message
    assert "uv >= 0.10.0" in message
    assert "Auto mode will use pip for this plan" in message
    mock_which.assert_called_once_with("uv")
    assert mock_run.call_count == 2
    assert [call.args[0] for call in mock_run.call_args_list] == [
        ["/usr/bin/uv", "--version"],
        PIP_VERSION_PROBE,
    ]


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_auto_uninstall_plan_uses_uv_remove_for_active_project(mock_which: Mock, tmp_path: Path) -> None:
    _write_project(tmp_path)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService(working_dir=tmp_path, active_virtualenv=True)

    plan = service.build_uninstall_plan(entry, catalog, manager="auto")

    assert plan.command == [
        "uv",
        "remove",
        "--project",
        str(tmp_path),
        "--no-sync",
        "data-designer-template",
    ]
    assert plan.commands == [
        [
            "uv",
            "remove",
            "--project",
            str(tmp_path),
            "--no-sync",
            "data-designer-template",
        ],
        [
            "uv",
            "pip",
            "uninstall",
            "--python",
            sys.executable,
            "data-designer-template",
        ],
    ]
    assert plan.uninstall_mode == "uv-project"
    assert plan.project_root == str(tmp_path)
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_uv_install_plan_targets_current_python_and_adds_catalog_index(mock_which: Mock) -> None:
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="uv")

    assert plan.command == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--constraint",
        "-",
        "--default-index",
        "https://pypi.org/simple/",
        "--index",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template",
    ]
    assert plan.command_stdin == (
        f"data-designer=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-config=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-engine=={DATA_DESIGNER_VERSION}\n"
    )
    assert plan.temporary_file is None


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_uv_install_plan_applies_version_specifier(mock_which: Mock) -> None:
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="uv", version_specifier="==0.1.0")

    assert plan.command == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--constraint",
        "-",
        "--default-index",
        "https://pypi.org/simple/",
        "--index",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template==0.1.0",
    ]
    assert plan.requirement == "data-designer-template==0.1.0"
    assert plan.command_stdin == (
        f"data-designer=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-config=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-engine=={DATA_DESIGNER_VERSION}\n"
    )
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_uv_add_plan_applies_version_specifier(mock_which: Mock, tmp_path: Path) -> None:
    _write_project(tmp_path)
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService(working_dir=tmp_path, active_virtualenv=True)

    plan = service.build_install_plan(entry, catalog, manager="uv", version_specifier="==0.1.0")

    assert plan.command == [
        "uv",
        "add",
        "--project",
        str(tmp_path),
        "--active",
        "--no-install-project",
        "--no-install-package",
        "data-designer",
        "--no-install-package",
        "data-designer-config",
        "--no-install-package",
        "data-designer-engine",
        "--index",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template==0.1.0",
    ]
    assert plan.requirement == "data-designer-template==0.1.0"
    assert plan.command_stdin is None
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_uv_add_plan_preserves_direct_reference_with_raw(mock_which: Mock, tmp_path: Path) -> None:
    _write_project(tmp_path)
    requirement = (
        "data-designer-template @ "
        "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@data-designer-template/v0.1.0"
    )
    entry = _entry(package_name="data-designer-template", install={"requirement": requirement})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService(working_dir=tmp_path, active_virtualenv=True)

    plan = service.build_install_plan(entry, catalog, manager="uv")

    assert plan.command[-2:] == ["--raw", requirement]
    assert "--index" not in plan.command
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value=None)
def test_build_uv_install_plan_raises_when_uv_is_unavailable(mock_which: Mock) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with pytest.raises(ValueError, match="uv was requested"):
        service.build_install_plan(entry, catalog, manager="uv")

    mock_which.assert_called_once_with("uv")


@patch(
    "data_designer.cli.services.plugin_install_service.subprocess.run",
    return_value=SimpleNamespace(returncode=1, stdout="", stderr="No module named pip"),
)
def test_build_pip_install_plan_raises_when_pip_is_unavailable(mock_run: Mock) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with pytest.raises(ValueError) as exc_info:
        service.build_install_plan(entry, catalog, manager="pip")

    message = str(exc_info.value)
    assert "pip was requested" in message
    assert "No module named pip" in message
    mock_run.assert_called_once()


@patch(
    "data_designer.cli.services.plugin_install_service.subprocess.run",
    return_value=SimpleNamespace(returncode=0, stdout="uv 0.7.22\n", stderr=""),
)
@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_uv_install_plan_raises_when_uv_is_too_old(
    mock_which: Mock,
    mock_run: Mock,
) -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with pytest.raises(ValueError) as exc_info:
        service.build_install_plan(entry, catalog, manager="uv")

    message = str(exc_info.value)
    assert "Found uv 0.7.22" in message
    assert "uv >= 0.10.0" in message
    assert "uv self update" in message
    assert "--manager pip" in message
    mock_which.assert_called_once_with("uv")
    mock_run.assert_called_once()


def test_build_install_plan_requires_installed_data_designer_version() -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with (
        patch(
            "data_designer.cli.services.plugin_install_service.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ),
        pytest.raises(ValueError, match="Unable to resolve installed 'data-designer' version"),
    ):
        service.build_install_plan(entry, catalog, manager="pip")


def test_build_install_plan_rejects_invalid_installed_data_designer_version() -> None:
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with (
        patch(
            "data_designer.cli.services.plugin_install_service.importlib.metadata.version",
            return_value="not a version",
        ),
        pytest.raises(ValueError, match="version 'not a version' is not a valid package version"),
    ):
        service.build_install_plan(entry, catalog, manager="pip")


def test_install_raises_when_runner_fails() -> None:
    service = PluginInstallService(runner=lambda command, stdin_text: 2)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    plan = service.build_install_plan(entry, catalog, manager="pip")

    with pytest.raises(RuntimeError, match="status 2"):
        service.install(plan)


def test_install_materializes_pip_constraint_as_temporary_file() -> None:
    seen: dict[str, Path | str | None] = {}

    def runner(command: list[str], stdin_text: str | None) -> int:
        constraint_file = Path(command[command.index("--constraint") + 1])
        seen["constraint_file"] = constraint_file
        seen["constraint_parent"] = constraint_file.parent
        seen["constraint_text"] = constraint_file.read_text(encoding="utf-8")
        seen["stdin_text"] = stdin_text
        return 0

    service = PluginInstallService(runner=runner)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    plan = service.build_install_plan(entry, catalog, manager="pip")

    service.install(plan)

    constraint_file = seen["constraint_file"]
    constraint_parent = seen["constraint_parent"]
    assert isinstance(constraint_file, Path)
    assert isinstance(constraint_parent, Path)
    assert not constraint_file.exists()
    assert not constraint_parent.exists()
    assert seen["constraint_text"] == (
        "# Data Designer is provided by the active CLI environment.\n"
        f"data-designer=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-config=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-engine=={DATA_DESIGNER_VERSION}\n"
    )
    assert seen["stdin_text"] is None


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_install_passes_uv_constraints_over_stdin(mock_which: Mock) -> None:
    seen: dict[str, list[str] | str | None] = {}

    def runner(command: list[str], stdin_text: str | None) -> int:
        seen["command"] = command
        seen["stdin_text"] = stdin_text
        return 0

    service = PluginInstallService(runner=runner)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    plan = service.build_install_plan(entry, catalog, manager="uv")

    service.install(plan)

    assert seen["command"] == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--constraint",
        "-",
        "data-designer-template",
    ]
    assert seen["stdin_text"] == (
        f"data-designer=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-config=={DATA_DESIGNER_VERSION}\n"
        f"data-designer-engine=={DATA_DESIGNER_VERSION}\n"
    )
    mock_which.assert_called_once_with("uv")


def test_uninstall_raises_when_runner_fails() -> None:
    service = PluginInstallService(runner=lambda command, stdin_text: 2)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    plan = service.build_uninstall_plan(entry, catalog, manager="pip")

    with pytest.raises(RuntimeError, match="status 2"):
        service.uninstall(plan)


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_uninstall_runs_every_project_uninstall_command(mock_which: Mock, tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def runner(command: list[str], stdin_text: str | None) -> int:
        assert stdin_text is None
        seen.append(command)
        return 0

    _write_project(tmp_path)
    service = PluginInstallService(runner=runner, working_dir=tmp_path, active_virtualenv=True)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    plan = service.build_uninstall_plan(entry, catalog, manager="auto")

    service.uninstall(plan)

    assert seen == plan.commands
    mock_which.assert_called_once_with("uv")


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
@patch("data_designer.cli.services.plugin_install_service.importlib.invalidate_caches")
def test_verify_entry_point_invalidates_caches_and_checks_declared_entry_point(
    mock_invalidate_caches: Mock,
    mock_entry_points: Mock,
) -> None:
    entry = _entry(
        package_name="data-designer-template",
        plugin_name="text-transform-v2",
        entry_point_name="text-transform",
        install={"requirement": "data-designer-template"},
    )
    mock_entry_points.return_value = [
        _installed_entry_point("other-plugin", "other_package.plugin:plugin"),
        _installed_entry_point("text-transform", "data_designer_template.plugin:plugin"),
    ]
    service = PluginInstallService()

    assert service.verify_entry_point(entry) is True
    mock_invalidate_caches.assert_called_once_with()
    mock_entry_points.assert_called_once_with(group="data_designer.plugins")


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_fails_when_name_matches_but_value_differs(mock_entry_points: Mock) -> None:
    entry = _entry(
        package_name="data-designer-template",
        plugin_name="text-transform",
        entry_point_name="text-transform",
        entry_point_value="data_designer_template.plugin:plugin",
        install={"requirement": "data-designer-template"},
    )
    mock_entry_points.return_value = [
        _installed_entry_point("text-transform", "other_package.plugin:plugin"),
    ]
    service = PluginInstallService()

    assert service.verify_entry_points([entry]) is False


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_succeeds_when_all_declared_entries_match(mock_entry_points: Mock) -> None:
    entries = [
        _entry(
            package_name="data-designer-template",
            plugin_name="text-transform",
            entry_point_name="text-transform",
            entry_point_value="data_designer_template.plugin:plugin",
            install={"requirement": "data-designer-template"},
        ),
        _entry(
            package_name="data-designer-profiler",
            plugin_name="text-profiler",
            entry_point_name="text-profiler",
            entry_point_value="data_designer_profiler.plugin:plugin",
            install={"requirement": "data-designer-profiler"},
        ),
    ]
    mock_entry_points.return_value = [
        _installed_entry_point("text-profiler", "data_designer_profiler.plugin:plugin"),
        _installed_entry_point("text-transform", "data_designer_template.plugin:plugin"),
    ]
    service = PluginInstallService()

    assert service.verify_entry_points(entries) is True


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_requires_every_declared_entry_point(mock_entry_points: Mock) -> None:
    entries = [
        _entry(
            package_name="data-designer-retrieval-sdg",
            plugin_name="document-chunker",
            entry_point_name="document-chunker",
            entry_point_value="data_designer_retrieval_sdg.chunker:plugin",
            install={"requirement": "data-designer-retrieval-sdg"},
        ),
        _entry(
            package_name="data-designer-retrieval-sdg",
            plugin_name="embedding-dedup",
            entry_point_name="embedding-dedup",
            entry_point_value="data_designer_retrieval_sdg.dedup:plugin",
            install={"requirement": "data-designer-retrieval-sdg"},
        ),
    ]
    mock_entry_points.return_value = [
        _installed_entry_point("document-chunker", "data_designer_retrieval_sdg.chunker:plugin")
    ]
    service = PluginInstallService()

    assert service.verify_entry_points(entries) is False


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_fails_when_matching_entry_point_cannot_load(
    mock_entry_points: Mock,
) -> None:
    entry = _entry(
        package_name="data-designer-template",
        plugin_name="smoke-broken",
        entry_point_name="smoke-broken",
        entry_point_value="missing_package.plugin:plugin",
        install={"requirement": "data-designer-template"},
    )
    mock_entry_points.return_value = [
        _installed_entry_point(
            "smoke-broken",
            "missing_package.plugin:plugin",
            load_side_effect=ModuleNotFoundError("No module named missing_package"),
        ),
    ]
    service = PluginInstallService()

    assert service.verify_entry_points([entry]) is False


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_fails_when_matching_entry_point_loads_non_plugin(mock_entry_points: Mock) -> None:
    entry = _entry(
        package_name="data-designer-template",
        plugin_name="smoke-broken",
        entry_point_name="smoke-broken",
        entry_point_value="data_designer_template.plugin:plugin",
        install={"requirement": "data-designer-template"},
    )
    mock_entry_points.return_value = [
        _installed_entry_point(
            "smoke-broken",
            "data_designer_template.plugin:plugin",
            load_result=object(),
        ),
    ]
    service = PluginInstallService()

    assert service.verify_entry_points([entry]) is False


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_verifies_multi_runtime_package_entries(mock_entry_points: Mock) -> None:
    entries = [
        _entry(
            package_name="data-designer-retrieval-sdg",
            plugin_name="document-chunker",
            entry_point_name="document-chunker",
            entry_point_value="data_designer_retrieval_sdg.chunker:plugin",
            install={"requirement": "data-designer-retrieval-sdg"},
        ),
        _entry(
            package_name="data-designer-retrieval-sdg",
            plugin_name="embedding-dedup",
            entry_point_name="embedding-dedup",
            entry_point_value="data_designer_retrieval_sdg.dedup:plugin",
            install={"requirement": "data-designer-retrieval-sdg"},
        ),
    ]
    mock_entry_points.return_value = [
        _installed_entry_point(
            name="embedding-dedup",
            value="data_designer_retrieval_sdg.dedup:plugin",
            package_name="data-designer-retrieval-sdg",
        ),
        _installed_entry_point(
            name="document-chunker",
            value="data_designer_retrieval_sdg.chunker:plugin",
            package_name="data-designer-retrieval-sdg",
        ),
    ]
    service = PluginInstallService()

    assert service.verify_entry_points(entries) is True


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
@patch("data_designer.cli.services.plugin_install_service.importlib.invalidate_caches")
def test_verify_entry_points_removed_succeeds_when_declared_entries_are_absent(
    mock_invalidate_caches: Mock,
    mock_entry_points: Mock,
) -> None:
    entry = _entry(
        package_name="data-designer-template",
        plugin_name="text-transform",
        entry_point_name="text-transform",
        entry_point_value="data_designer_template.plugin:plugin",
        install={"requirement": "data-designer-template"},
    )
    mock_entry_points.return_value = [
        SimpleNamespace(name="other-plugin", value="other_package.plugin:plugin"),
    ]
    service = PluginInstallService()

    assert service.verify_entry_points_removed([entry]) is True
    mock_invalidate_caches.assert_called_once_with()
    mock_entry_points.assert_called_once_with(group="data_designer.plugins")


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_removed_fails_when_declared_entry_still_exists(mock_entry_points: Mock) -> None:
    entry = _entry(
        package_name="data-designer-template",
        plugin_name="text-transform",
        entry_point_name="text-transform",
        entry_point_value="data_designer_template.plugin:plugin",
        install={"requirement": "data-designer-template"},
    )
    mock_entry_points.return_value = [
        SimpleNamespace(name="text-transform", value="data_designer_template.plugin:plugin"),
    ]
    service = PluginInstallService()

    assert service.verify_entry_points_removed([entry]) is False


def _installed_entry_point(
    name: str,
    value: str,
    *,
    package_name: str | None = None,
    load_result: object | None = None,
    load_side_effect: BaseException | None = None,
) -> SimpleNamespace:
    def load() -> object:
        if load_side_effect is not None:
            raise load_side_effect
        return load_result if load_result is not None else _loaded_plugin()

    entry_point = SimpleNamespace(name=name, value=value, load=load)
    if package_name is not None:
        entry_point.dist = SimpleNamespace(metadata={"Name": package_name})
    return entry_point


def _loaded_plugin() -> Plugin:
    return Plugin.model_construct(
        impl_qualified_name="data_designer_template.plugin.TextTransform",
        config_qualified_name="data_designer_template.plugin.TextTransformConfig",
        plugin_type=PluginType.PROCESSOR,
    )


def _entry(
    *,
    package_name: str,
    install: dict,
    plugin_name: str = "text-transform",
    entry_point_name: str = "text-transform",
    entry_point_value: str = "data_designer_template.plugin:plugin",
) -> PluginCatalogEntry:
    payload = {
        "name": plugin_name,
        "plugin_type": "processor",
        "description": "Transform text records",
        "package": {
            "name": package_name,
        },
        "install": install,
        "entry_point": {
            "group": "data_designer.plugins",
            "name": entry_point_name,
            "value": entry_point_value,
        },
        "compatibility": {
            "python": {"specifier": ">=3.10"},
            "data_designer": {
                "requirement": "data-designer>=0.5.7",
                "specifier": ">=0.5.7",
                "marker": None,
            },
        },
        "docs": {
            "url": f"https://docs.example.test/plugins/{package_name}/",
        },
    }
    return PluginCatalogEntry.model_validate(payload)


def _write_project(path: Path, *, name: str = "synthetic-data-project") -> Path:
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n', encoding="utf-8")
    return path
