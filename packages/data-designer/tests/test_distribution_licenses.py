# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import io
import tarfile
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location("package_install_checks", ROOT / "scripts/test_slurm_package_install.py")
assert SPEC and SPEC.loader
PACKAGE_CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE_CHECKS)


@pytest.mark.parametrize("package", PACKAGE_CHECKS.PACKAGE_PATHS)
def test_package_license_copy_matches_repository(package: str) -> None:
    directory = ROOT / package
    assert (directory / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    metadata = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
    assert metadata["license-files"] == ["LICENSE"]


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("license_state", ["valid", "missing", "altered", "missing_metadata", "wrong_expression"])
def test_distribution_requires_full_license_text(tmp_path: Path, kind: str, license_state: str) -> None:
    license_text = (ROOT / "LICENSE").read_bytes()
    metadata = b"Name: example\nVersion: 1.0\nLicense-Expression: Apache-2.0\nLicense-File: LICENSE\n"
    if license_state == "missing_metadata":
        metadata = metadata.replace(b"License-File: LICENSE\n", b"")
    elif license_state == "wrong_expression":
        metadata = metadata.replace(b"Apache-2.0", b"MIT")
    members = {}
    if license_state != "missing":
        members["LICENSE"] = b"Apache-2.0\n" if license_state == "altered" else license_text
    if kind == "wheel":
        path = tmp_path / "example-1.0-py3-none-any.whl"
        with ZipFile(path, "w") as archive:
            archive.writestr("example-1.0.dist-info/METADATA", metadata)
            for name, content in members.items():
                archive.writestr(f"example-1.0.dist-info/licenses/{name}", content)
    else:
        path = tmp_path / "example-1.0.tar.gz"
        members["PKG-INFO"] = metadata
        with tarfile.open(path, "w:gz") as archive:
            for name, content in members.items():
                info = tarfile.TarInfo(f"example-1.0/{name}")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    if license_state == "valid":
        PACKAGE_CHECKS.verify_package_license(path)
    else:
        with pytest.raises(
            AssertionError, match="metadata" if license_state in {"missing_metadata", "wrong_expression"} else "LICENSE"
        ):
            PACKAGE_CHECKS.verify_package_license(path)
