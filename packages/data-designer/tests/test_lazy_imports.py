# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _run_python_snippet(code: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Python snippet failed:\n{result.stdout}\n{result.stderr}"
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_lazy_heavy_imports_defer_pandas_until_attribute_access() -> None:
    lines = _run_python_snippet(
        "\n".join(
            [
                "import sys",
                "import data_designer.lazy_heavy_imports as lazy",
                "print(int('pandas' in sys.modules))",
                "_ = lazy.__dir__",
                "print(int('pandas' in sys.modules))",
                "_ = lazy.pd",
                "print(int('pandas' in sys.modules))",
            ]
        )
    )
    assert lines == ["0", "0", "1"]


def test_importing_config_module_does_not_eagerly_import_pandas() -> None:
    lines = _run_python_snippet(
        "\n".join(
            [
                "import sys",
                "import data_designer.config as dd",
                "print(int('pandas' in sys.modules))",
                "_ = dd.__all__",
                "print(int('pandas' in sys.modules))",
            ]
        )
    )
    assert lines == ["0", "0"]


def test_runtime_src_avoids_from_lazy_heavy_imports_pattern() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pattern = re.compile(r"^\s*from\s+data_designer\.lazy_heavy_imports\s+import\b", re.MULTILINE)
    offenders: list[str] = []

    for path in sorted(repo_root.glob("packages/*/src/**/*.py")):
        if path.name == "lazy_heavy_imports.py":
            continue
        if pattern.search(path.read_text()):
            offenders.append(str(path.relative_to(repo_root)))

    assert not offenders, f"Runtime source files should avoid from-import lazy pattern: {offenders}"
