# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOCKET_CALLS: list[dict[str, Any]] = []
TEST_DURATIONS: dict[str, float] = defaultdict(float)


def pytest_configure() -> None:
    for name in ("getfqdn", "getaddrinfo"):
        original = getattr(socket, name)

        def timed(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                if (
                    _name == "getfqdn"
                    and os.environ.get("DD_SMOKE_NUMERIC_LOOPBACK") == "1"
                    and args
                    and args[0] in ("127.0.0.1", "::1")
                ):
                    return args[0]
                return _original(*args, **kwargs)
            finally:
                SOCKET_CALLS.append(
                    {
                        "function": _name,
                        "address": str(args[0]) if args else "",
                        "seconds": time.perf_counter() - started,
                    }
                )

        setattr(socket, name, timed)


def pytest_runtest_logreport(report: Any) -> None:
    TEST_DURATIONS[report.nodeid] += report.duration


def pytest_sessionfinish() -> None:
    result = {
        "socket_calls_in_pytest_process": SOCKET_CALLS,
        "test_seconds_including_setup_teardown": TEST_DURATIONS,
    }
    Path(os.environ["DD_SMOKE_REPORT"]).write_text(json.dumps(result, indent=2) + "\n")
    print(
        "\nSOCKET SUMMARY",
        json.dumps(
            {
                name: sum(c["seconds"] for c in SOCKET_CALLS if c["function"] == name)
                for name in ("getfqdn", "getaddrinfo")
            }
        ),
    )


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def license_smoke() -> None:
    module = load_module("license_smoke", ROOT / "scripts/update_license_headers.py")
    folders = [ROOT / "scripts", ROOT / "tests_e2e"]
    folders.extend(
        p / sub for p in (ROOT / "packages").iterdir() if p.is_dir() for sub in ("src", "tests", "dev-tools")
    )
    started = time.perf_counter()
    with patch.object(module.subprocess, "run", side_effect=AssertionError("license check requested Git history")):
        results = [module.main(folder, check_only=True) for folder in folders if folder.exists()]
    failures = [str(path) for result in results for path in result[3]]
    print(
        json.dumps(
            {
                "files": sum(r[0] for r in results),
                "git_calls": 0,
                "seconds": time.perf_counter() - started,
                "failures": failures,
            }
        )
    )
    assert not failures
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "missing.py"
        missing.write_text("print('missing header')\n")
        with patch.object(module, "get_file_creation_year", return_value=None):
            assert module.main(Path(directory), check_only=True)[1] == 1


def recipe_smoke() -> None:
    module = load_module("recipe_smoke", ROOT / "docs/assets/recipes/workflow_chaining/document_review_gate.py")
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        started = time.perf_counter()
        for index in range(5):
            module.generate_sample_pages(base / f"repeated-{index}", count=4, seed=3)
        repeated = time.perf_counter() - started
        started = time.perf_counter()
        pages = module.generate_sample_pages(base / "shared", count=4, seed=3)
        for _ in range(5):
            copy = pages.copy(deep=True)
            module.validate_metadata_rows(copy)
            copy.loc[0, "ground_truth_boxes"] = "[]"
            assert pages.loc[0, "ground_truth_boxes"] != "[]"
        shared = time.perf_counter() - started
    print(json.dumps({"five_generations_seconds": repeated, "one_generation_five_copies_seconds": shared}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=("license", "recipe"))
    args = parser.parse_args()
    if args.probe == "license":
        license_smoke()
    else:
        recipe_smoke()
