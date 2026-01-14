#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert MkDocs admonitions to Fern callouts."""
import re
import sys

ADMONITION_MAP = {
    "note": "Note",
    "tip": "Tip",
    "info": "Info",
    "warning": "Warning",
    "danger": "Warning",
    "question": "Info",
    "example": "Info",
    "abstract": "Note",
    "success": "Tip",
    "failure": "Warning",
    "bug": "Warning",
}


def convert_admonitions(content: str) -> str:
    """Convert !!! admonitions to <Callout> components."""
    pattern = r'!!! (\w+)(?: "([^"]*)")?\n((?:    .*\n?)*)'

    def replace(match: re.Match) -> str:
        admon_type = match.group(1).lower()
        title = match.group(2) or ""
        body = match.group(3)
        # Remove 4-space indent from body
        body = re.sub(r"^    ", "", body, flags=re.MULTILINE).strip()
        fern_type = ADMONITION_MAP.get(admon_type, "Note")
        if title:
            return f'<{fern_type} title="{title}">\n{body}\n</{fern_type}>\n'
        return f"<{fern_type}>\n{body}\n</{fern_type}>\n"

    return re.sub(pattern, replace, content)


if __name__ == "__main__":
    content = sys.stdin.read()
    print(convert_admonitions(content))
