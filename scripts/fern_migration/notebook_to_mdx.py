#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Jupyter notebook source (.py format) to MDX."""
import re
import sys
from pathlib import Path


def notebook_py_to_mdx(notebook_path: str, colab_url: str, title: str | None = None) -> str:
    """Convert a Jupyter notebook source file (.py with Jupytext format) to MDX format."""
    with open(notebook_path) as f:
        content = f.read()

    # Extract title from the notebook if not provided
    if title is None:
        title_match = re.search(r"# # (.+)", content)
        if title_match:
            title = title_match.group(1).strip()
            # Remove emoji if present
            title = re.sub(r"^[🎨📓🏥]\s*", "", title)
        else:
            title = Path(notebook_path).stem.replace("-", " ").title()

    lines = [
        "---",
        f"title: {title}",
        "---",
        "",
        '<Info title="Interactive Version">',
        f"Run this tutorial interactively in [Google Colab]({colab_url}).",
        "</Info>",
        "",
    ]

    # Process the notebook content
    in_markdown_block = False
    in_code_block = False
    current_content = []

    for line in content.split("\n"):
        # Skip Jupytext header
        if line.startswith("# ---") or line.startswith("#   "):
            continue

        # Markdown cell marker
        if line == "# %% [markdown]":
            if in_code_block:
                lines.append("```")
                lines.append("")
                in_code_block = False
            in_markdown_block = True
            continue

        # Code cell marker
        if line == "# %%":
            if in_markdown_block:
                in_markdown_block = False
            if in_code_block:
                lines.append("```")
                lines.append("")
            lines.append("```python")
            in_code_block = True
            continue

        # Process content
        if in_markdown_block:
            # Remove the '# ' prefix from markdown lines
            if line.startswith("# "):
                lines.append(line[2:])
            elif line == "#":
                lines.append("")
            else:
                lines.append(line)
        elif in_code_block:
            lines.append(line)

    # Close any open code block
    if in_code_block:
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: notebook_to_mdx.py <notebook.py> <colab_url> [title]")
        sys.exit(1)
    title = sys.argv[3] if len(sys.argv) > 3 else None
    print(notebook_py_to_mdx(sys.argv[1], sys.argv[2], title))
