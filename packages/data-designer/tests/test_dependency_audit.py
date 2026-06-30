# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "audit_package_dependencies.py"
SPEC = importlib.util.spec_from_file_location("audit_package_dependencies", SCRIPT_PATH)
assert SPEC and SPEC.loader
DEPENDENCY_AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPENDENCY_AUDIT)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    config = tmp_path / "packages" / "data-designer-config"
    engine = tmp_path / "packages" / "data-designer-engine"
    (config / "src").mkdir(parents=True)
    (engine / "src").mkdir(parents=True)
    (config / "pyproject.toml").write_text(
        '[project]\nname = "data-designer-config"\ndependencies = ["jinja2>=3.1.6,<4"]\n'
    )
    (engine / "pyproject.toml").write_text(
        """
[project]
name = "data-designer-engine"
dynamic = ["dependencies"]

[tool.hatch.metadata.hooks.uv-dynamic-versioning]
dependencies = ["data-designer-config=={{ version }}"]
""".lstrip()
    )
    (engine / "src" / "engine.py").write_text(
        "from __future__ import annotations\n\nimport os\nfrom jinja2 import Template\n"
    )
    return tmp_path


def audit(
    workspace: Path,
    distributions: dict[str, list[str]],
    requirements: dict[str, list[str]] | None = None,
) -> dict:
    module = DEPENDENCY_AUDIT
    assert isinstance(module, ModuleType)
    return module.audit_repository(workspace, distributions, requirements)


def test_marks_transitively_guaranteed_gap_low(workspace: Path) -> None:
    result = audit(workspace, {"jinja2": ["Jinja2"]})

    engine = next(package for package in result["packages"] if package["package"] == "data-designer-engine")
    assert engine["missing"] == [
        {
            "dependency": "jinja2",
            "modules": ["jinja2"],
            "files": ["packages/data-designer-engine/src/engine.py"],
            "declared_by": ["data-designer-config"],
            "guaranteed_by": ["data-designer-config"],
            "severity": "low",
        }
    ]


def test_marks_uncovered_gap_high(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "example"
    (package / "src").mkdir(parents=True)
    (package / "pyproject.toml").write_text('[project]\nname = "example"\ndependencies = []\n')
    (package / "src" / "example.py").write_text("import httpx\n")

    result = audit(tmp_path, {"httpx": ["httpx"]})

    assert result["packages"][0]["missing"][0]["severity"] == "high"
    assert result["packages"][0]["missing"][0]["guaranteed_by"] == []


def test_marks_external_transitive_gap_low(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "example"
    (package / "src").mkdir(parents=True)
    (package / "pyproject.toml").write_text('[project]\nname = "example"\ndependencies = ["typer"]\n')
    (package / "src" / "example.py").write_text("import click\n")

    result = audit(tmp_path, {"click": ["click"]}, {"typer": ["click"]})

    assert result["packages"][0]["missing"][0]["severity"] == "low"
    assert result["packages"][0]["missing"][0]["guaranteed_by"] == ["typer"]
