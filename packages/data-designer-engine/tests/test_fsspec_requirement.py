# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement

_ENGINE_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_DYNAMIC_DEPENDENCIES = re.compile(
    r"\[tool\.hatch\.metadata\.hooks\.uv-dynamic-versioning\]\s*dependencies = \[(.*?)\n\]",
    re.DOTALL,
)
_QUOTED_REQUIREMENT = re.compile(r'"([^"]+)"')
_FSSPEC_RELEASE_FIXING_CVE_2026_104851 = "2026.6.0"


def test_engine_accepts_fsspec_release_that_fixes_cve_2026_104851() -> None:
    requirement = Requirement(_declared_fsspec_requirement(_ENGINE_PYPROJECT.read_text(encoding="utf-8")))

    assert requirement.name == "fsspec"
    assert requirement.specifier.contains(_FSSPEC_RELEASE_FIXING_CVE_2026_104851)
    assert requirement.specifier.contains("2025.3.0")
    assert not requirement.specifier.contains("2025.2.0")
    assert not requirement.specifier.contains("2027.0.0")


def _declared_fsspec_requirement(pyproject_text: str) -> str:
    """Return the fsspec requirement published by the engine metadata hook."""
    hook_match = _DYNAMIC_DEPENDENCIES.search(pyproject_text)
    if hook_match is None:
        raise AssertionError("engine metadata hook does not declare dependencies")

    fsspec_requirements = [
        requirement
        for requirement in _QUOTED_REQUIREMENT.findall(hook_match.group(1))
        if requirement.startswith("fsspec")
    ]
    if len(fsspec_requirements) != 1:
        raise AssertionError(f"expected one fsspec requirement, found {fsspec_requirements}")
    return fsspec_requirements[0]
