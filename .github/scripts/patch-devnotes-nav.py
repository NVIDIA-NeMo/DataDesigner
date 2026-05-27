# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch the Dev Notes nav block in mkdocs.yml.

Used by publish-devnotes.yml to splice HEAD's Dev Notes nav entries into an
older source checkout without touching the rest of the file.

Usage: python patch-devnotes-nav.py <head_mkdocs> <target_mkdocs>
"""

from __future__ import annotations

import re
import sys


def extract_devnotes_block(text: str) -> tuple[int, int, list[str]]:
    """Return (start, end, lines) for the '  - Dev Notes:' nav block."""
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^  - Dev Notes:", line):
            start = i
            break
    if start is None:
        raise SystemExit("Dev Notes nav section not found")
    end = start + 1
    while end < len(lines):
        # Stop at next top-level nav entry (2-space indent) or non-nav section
        if lines[end].strip() and not lines[end].startswith("      ") and not lines[end].startswith("      #"):
            break
        end += 1
    return start, end, lines


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} <head_mkdocs> <target_mkdocs>")

    head_path, target_path = sys.argv[1], sys.argv[2]

    with open(head_path) as f:
        head_start, head_end, head_lines = extract_devnotes_block(f.read())
    head_block = head_lines[head_start:head_end]

    with open(target_path) as f:
        old_start, old_end, old_lines = extract_devnotes_block(f.read())
    new_lines = old_lines[:old_start] + head_block + old_lines[old_end:]

    with open(target_path, "w") as f:
        f.writelines(new_lines)
    print(f"Patched Dev Notes nav: replaced lines {old_start + 1}-{old_end} with {len(head_block)} lines from HEAD")


if __name__ == "__main__":
    main()
