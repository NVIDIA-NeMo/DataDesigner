# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.cli.plugin_catalog import parse_plugin_package_install_request


@pytest.mark.parametrize(
    ("package", "version", "expected_package", "expected_specifier"),
    [
        ("github", None, "github", None),
        ("github==0.1.0", None, "github", "==0.1.0"),
        ("data-designer-github", "0.1.0", "data-designer-github", "==0.1.0"),
        ("github", "v0.1.0", "github", "==0.1.0"),
    ],
)
def test_parse_plugin_package_install_request_accepts_names_aliases_and_versions(
    package: str,
    version: str | None,
    expected_package: str,
    expected_specifier: str | None,
) -> None:
    request = parse_plugin_package_install_request(package, version=version)

    assert request.package == expected_package
    assert request.version_specifier == expected_specifier


@pytest.mark.parametrize(
    "package",
    [
        "github[extra]",
        'github; python_version >= "3.10"',
        "github @ https://packages.example.test/data_designer_github-0.1.0-py3-none-any.whl",
    ],
)
def test_parse_plugin_package_install_request_rejects_non_catalog_targets(package: str) -> None:
    with pytest.raises(ValueError, match="catalog package names or package aliases"):
        parse_plugin_package_install_request(package)


def test_parse_plugin_package_install_request_rejects_invalid_package_text() -> None:
    with pytest.raises(ValueError, match="Invalid plugin package"):
        parse_plugin_package_install_request("not a requirement !!!")


def test_parse_plugin_package_install_request_rejects_conflicting_version_inputs() -> None:
    with pytest.raises(ValueError, match="either in PACKAGE or with --version"):
        parse_plugin_package_install_request("github==0.1.0", version="0.1.1")


def test_parse_plugin_package_install_request_rejects_invalid_version_option() -> None:
    with pytest.raises(ValueError, match="Invalid plugin package version"):
        parse_plugin_package_install_request("github", version="not-a-version")
