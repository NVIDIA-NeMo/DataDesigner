# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
SPEC = importlib.util.spec_from_file_location("license_header_checks", ROOT / "scripts/update_license_headers.py")
assert SPEC and SPEC.loader
HEADER_CHECKS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HEADER_CHECKS
SPEC.loader.exec_module(HEADER_CHECKS)
HEADER = HEADER_CHECKS.generate_license_header("2026")


@pytest.mark.parametrize("existing", ["", HEADER_CHECKS.generate_license_header("2025"), HEADER])
def test_header_updates_preserve_jupytext_front_matter(tmp_path: Path, existing: str) -> None:
    metadata = "# ---\n# jupyter:\n#   kernelspec:\n#     name: python3\n# ---\n\n"
    body = "# %% [markdown]\n# Tutorial\n\n# %%\nprint('hello')\n"
    path = tmp_path / "tutorial.py"
    path.write_text(metadata + existing + body)

    HEADER_CHECKS.update_license_header_in_file(path, HEADER)

    assert path.read_text() == metadata + HEADER + body
    assert HEADER_CHECKS.check_license_header_matches(path, HEADER) == (True, "match")
    assert HEADER_CHECKS.update_license_header_in_file(path, HEADER) == (False, "unchanged")


@pytest.mark.parametrize("source", ["# ---\n# ordinary comment\n# ---\n", "# ---\n# jupyter:\n"])
def test_non_jupytext_or_incomplete_front_matter_is_not_skipped(tmp_path: Path, source: str) -> None:
    path = tmp_path / "ordinary.py"
    path.write_text(source)

    HEADER_CHECKS.update_license_header_in_file(path, HEADER)

    assert path.read_text() == HEADER + source


def test_empty_notebook_keeps_its_metadata_when_adding_a_header(tmp_path: Path) -> None:
    metadata = "# ---\n# jupyter:\n#   kernelspec:\n#     name: python3\n# ---\n\n"
    path = tmp_path / "empty.py"
    path.write_text(metadata)

    HEADER_CHECKS.update_license_header_in_file(path, HEADER)

    assert path.read_text() == metadata + HEADER.rstrip() + "\n"
    assert HEADER_CHECKS.check_license_header_matches(path, HEADER) == (True, "match")
