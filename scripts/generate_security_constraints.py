# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export reviewed uv security floors as pip-compatible constraints."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = """# Security version floors generated from [tool.uv].constraint-dependencies in pyproject.toml.
# Regenerate with: make security-constraints
# Use with: pip install data-designer -c constraints.txt
#           uv pip install data-designer -c constraints.txt
#           uv add data-designer -c constraints.txt
"""


def generate_constraints(root: Path, *, check: bool = False) -> int:
    """Export uv security constraints, or fail if the committed export is stale."""
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    constraints = config["tool"]["uv"]["constraint-dependencies"]
    content = HEADER + "\n".join(sorted(constraints, key=str.casefold)) + "\n"
    destination = root / "constraints.txt"
    if check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != content:
            print("constraints.txt is out of date. Run make security-constraints.")
            return 1
        print("constraints.txt matches the uv security constraints.")
    else:
        destination.write_text(content, encoding="utf-8")
        print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without rewriting constraints.txt")
    raise SystemExit(generate_constraints(ROOT, check=parser.parse_args().check))
