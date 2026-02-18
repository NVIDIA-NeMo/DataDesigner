#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert MkDocs tabs to Fern Tabs components."""
import re
import sys


def convert_tabs(content: str) -> str:
    """Convert === tabs to <Tabs> components."""
    # Match tab groups
    pattern = r'((?:=== "([^"]+)"\n((?:    .*\n?)*)\n?)+)'

    def replace_group(match: re.Match) -> str:
        group = match.group(0)
        tabs = re.findall(r'=== "([^"]+)"\n((?:    .*\n?)*)', group)
        result = ["<Tabs>"]
        for title, body in tabs:
            body = re.sub(r"^    ", "", body, flags=re.MULTILINE).strip()
            # Indent the body content properly
            body_lines = body.split("\n")
            indented_body = "\n".join(["    " + line if line.strip() else "" for line in body_lines])
            result.append(f'  <Tab title="{title}">')
            result.append(indented_body)
            result.append("  </Tab>")
        result.append("</Tabs>")
        return "\n".join(result) + "\n"

    return re.sub(pattern, replace_group, content)


if __name__ == "__main__":
    content = sys.stdin.read()
    print(convert_tabs(content))
