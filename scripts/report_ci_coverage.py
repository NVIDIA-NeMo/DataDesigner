# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from coverage import Coverage, CoverageData


def report_coverage() -> float:
    paths = [Path(f".coverage.{package}") for package in ("config", "engine", "interface")]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing coverage data: {path}")
        data = CoverageData(basename=str(path))
        data.read()
        if not data.measured_files():
            raise ValueError(f"Empty coverage data: {path}")

    coverage = Coverage()
    coverage.combine(data_paths=[str(path) for path in paths], strict=True, keep=True)
    coverage.save()
    percentage = coverage.report(show_missing=True)
    coverage.xml_report()
    if percentage < 90:
        raise ValueError(f"Coverage {percentage:.2f}% is below 90%")
    return percentage


if __name__ == "__main__":
    report_coverage()
