# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPOSITORY_ROOT = Path(__file__).parents[1]
PACKAGE_PATHS = (
    "packages/data-designer-config",
    "packages/data-designer-engine",
    "packages/data-designer",
    "packages/data-designer-slurm",
)


def run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def wheel_metadata(path: Path) -> Message:
    with ZipFile(path) as wheel:
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        return BytesParser().parsebytes(wheel.read(metadata_path))


def build_wheels(uv: str, wheel_directory: Path) -> dict[str, Path]:
    for package_path in PACKAGE_PATHS:
        run(
            [uv, "build", "--wheel", "--out-dir", str(wheel_directory), package_path],
            cwd=REPOSITORY_ROOT,
        )

    wheels = {}
    for wheel_path in wheel_directory.glob("*.whl"):
        name = canonicalize_name(wheel_metadata(wheel_path)["Name"])
        wheels[name] = wheel_path
    return wheels


def requirement(metadata: Message, dependency_name: str) -> Requirement:
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    return next(item for item in requirements if canonicalize_name(item.name) == dependency_name)


def python_path(environment_path: Path) -> Path:
    if os.name == "nt":
        return environment_path / "Scripts" / "python.exe"
    return environment_path / "bin" / "python"


def create_environment(uv: str, environment_path: Path, *, cwd: Path) -> Path:
    run([uv, "venv", "--python", sys.executable, str(environment_path)], cwd=cwd)
    return python_path(environment_path)


def install(uv: str, python: Path, wheel_directory: Path, package: str, *, cwd: Path) -> None:
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--prerelease=allow",
            "--find-links",
            str(wheel_directory),
            package,
        ],
        cwd=cwd,
    )


def verify_install(python: Path, version: str, *, slurm: bool, cwd: Path) -> None:
    statement = f"""
from importlib.metadata import version
from importlib.util import find_spec

import data_designer
import data_designer.config
import data_designer.engine
import data_designer.interface

assert data_designer.__file__ is None
assert version("data-designer") == {version!r}
assert (find_spec("data_designer.slurm") is not None) is {slurm!r}
"""
    if slurm:
        statement += f'\nimport data_designer.slurm\nassert version("data-designer-slurm") == {version!r}\n'
    run([str(python), "-c", statement], cwd=cwd)


def main() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        wheel_directory = root / "wheels"
        wheel_directory.mkdir()
        wheels = build_wheels(uv, wheel_directory)

        base_wheel = wheels["data-designer"]
        leaf_wheel = wheels["data-designer-slurm"]
        base_metadata = wheel_metadata(base_wheel)
        leaf_metadata = wheel_metadata(leaf_wheel)
        version = base_metadata["Version"]
        assert leaf_metadata["Version"] == version

        base_leaf_requirement = requirement(base_metadata, "data-designer-slurm")
        leaf_base_requirement = requirement(leaf_metadata, "data-designer")
        assert str(base_leaf_requirement.specifier) == f"=={version}"
        assert str(leaf_base_requirement.specifier) == f"=={version}"
        assert base_leaf_requirement.marker is not None
        assert base_leaf_requirement.marker.evaluate({"extra": "slurm"})
        assert not base_leaf_requirement.marker.evaluate({"extra": ""})
        assert "slurm" in base_metadata.get_all("Provides-Extra", [])
        with ZipFile(leaf_wheel) as wheel:
            assert "data_designer/__init__.py" not in wheel.namelist()

        base_python = create_environment(uv, root / "base", cwd=root)
        install(uv, base_python, wheel_directory, f"data-designer=={version}", cwd=root)
        verify_install(base_python, version, slurm=False, cwd=root)

        extra_python = create_environment(uv, root / "extra", cwd=root)
        install(uv, extra_python, wheel_directory, f"data-designer[slurm]=={version}", cwd=root)
        verify_install(extra_python, version, slurm=True, cwd=root)

        leaf_python = create_environment(uv, root / "leaf", cwd=root)
        install(uv, leaf_python, wheel_directory, f"data-designer-slurm=={version}", cwd=root)
        verify_install(leaf_python, version, slurm=True, cwd=root)

        leaf_only_directory = root / "leaf-only"
        leaf_only_directory.mkdir()
        shutil.copy2(leaf_wheel, leaf_only_directory)
        missing_python = create_environment(uv, root / "missing", cwd=root)
        result = run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(missing_python),
                "--no-index",
                "--find-links",
                str(leaf_only_directory),
                str(leaf_wheel),
            ],
            cwd=root,
            check=False,
        )
        assert result.returncode != 0
        assert "data-designer" in (result.stdout + result.stderr)


if __name__ == "__main__":
    main()
