# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


def redact_output(text: str) -> str:
    """Remove configured credentials and routing information from CI output."""
    names = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "AGENTIC_CI_MODEL", "GH_TOKEN")
    values = {os.environ[name] for name in names if os.environ.get(name)}
    host = urlsplit(os.environ.get("ANTHROPIC_BASE_URL", "")).hostname
    if host:
        values.add(host)
    if not values:
        return text

    pattern = "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))
    return re.sub(pattern, "[redacted]", text, flags=re.IGNORECASE)


def main() -> None:
    for filename in sys.argv[1:]:
        path = Path(filename)
        if path.is_file():
            path.write_text(redact_output(path.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
