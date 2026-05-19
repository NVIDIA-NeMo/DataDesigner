#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert py2fern self-named module pages into Fern folder overview pages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def normalize_root(root: Path) -> int:
    renamed = 0
    for path in sorted(root.rglob("*.mdx")):
        if path.name == "index.mdx":
            continue
        if normalized(path.stem) != normalized(path.parent.name):
            continue

        target = path.with_name("index.mdx")
        if target.exists():
            raise FileExistsError(f"Cannot rename {path}: {target} already exists")
        path.rename(target)
        renamed += 1
    return renamed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="py2fern output roots to normalize")
    args = parser.parse_args()

    count = 0
    for root in args.roots:
        if not root.exists():
            raise FileNotFoundError(root)
        count += normalize_root(root)
    print(f"Normalized {count} py2fern pages to index.mdx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
