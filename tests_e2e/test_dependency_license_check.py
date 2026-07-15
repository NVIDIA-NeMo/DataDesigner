# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_dependency_licenses.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dependency_licenses", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_report_accepts_permissive_and_composite_licenses() -> None:
    module = load_script()
    policy = module.LicensePolicy(
        allowed_licenses=frozenset({"Apache-2.0", "BSD-3-Clause", "MIT"}),
        exceptions={},
    )
    packages = [
        module.PackageLicense("apache", "1", "Apache-2.0", ""),
        module.PackageLicense("dual", "1", "Apache-2.0 OR BSD-3-Clause", ""),
        module.PackageLicense("mit", "1", "MIT License", ""),
    ]

    violations, reviewed = module.evaluate_report(packages, policy)

    assert violations == []
    assert reviewed == []


def test_evaluate_report_rejects_copyleft_and_unknown_licenses() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={})
    packages = [
        module.PackageLicense("copyleft", "1", "LGPL-2.1-or-later", ""),
        module.PackageLicense("unknown", "1", "UNKNOWN", ""),
    ]

    violations, _ = module.evaluate_report(packages, policy)

    assert violations == [
        "copyleft==1: disallowed license(s): LGPL-2.1-or-later",
        "unknown==1: unknown or unrecognized license 'UNKNOWN'",
    ]


def test_evaluate_report_recognizes_bundled_mit_license_text() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"MIT"}), exceptions={})
    package = module.PackageLicense(
        "missing-metadata",
        "1",
        "UNKNOWN",
        "MIT License\n\nCopyright example\n\nPermission is hereby granted, free of charge, to any person obtaining a copy",
    )

    violations, _ = module.evaluate_report([package], policy)

    assert violations == []


def test_evaluate_report_requires_exception_license_to_remain_unchanged() -> None:
    module = load_script()
    exception = module.ExceptionPolicy(license="MPL-2.0", reason="Reviewed separately.")
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={"example": exception})
    package = module.PackageLicense("example", "2", "GPL-3.0-only", "")

    violations, reviewed = module.evaluate_report([package], policy)

    assert violations == ["example==2: exception expected 'MPL-2.0', but package reports 'GPL-3.0-only'"]
    assert reviewed == []


def test_evaluate_report_rejects_stale_exceptions() -> None:
    module = load_script()
    exception = module.ExceptionPolicy(license="MPL-2.0", reason="Reviewed separately.")
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={"removed": exception})

    violations, _ = module.evaluate_report([], policy)

    assert violations == ["removed: stale exception; package is not present in the scanned environment"]
