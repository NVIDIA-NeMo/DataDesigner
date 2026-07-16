# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_dependency_licenses.py"
MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2026 Example

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


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


def test_evaluate_report_accepts_allowed_or_alternative() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0", "MIT"}), exceptions={})
    packages = [
        module.PackageLicense("dual", "1", "MIT OR GPL-3.0-only", ""),
        module.PackageLicense("grouped", "1", "Apache-2.0 AND (MIT OR GPL-3.0-only)", ""),
    ]

    violations, _ = module.evaluate_report(packages, policy)

    assert violations == []


def test_evaluate_report_requires_all_and_terms() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={})
    package = module.PackageLicense("conjunctive", "1", "Apache-2.0 AND GPL-3.0-only", "")

    violations, _ = module.evaluate_report([package], policy)

    assert violations == ["conjunctive==1: disallowed license(s): GPL-3.0-only"]


def test_evaluate_report_rejects_disallowed_or_alternatives_and_malformed_expressions() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={})
    packages = [
        module.PackageLicense("disallowed", "1", "GPL-2.0-only OR GPL-3.0-only", ""),
        module.PackageLicense("malformed", "1", "Apache-2.0 OR", ""),
    ]

    violations, _ = module.evaluate_report(packages, policy)

    assert violations == [
        "disallowed==1: disallowed license(s): GPL-2.0-only, GPL-3.0-only",
        "malformed==1: unknown or unrecognized license 'Apache-2.0 OR'",
    ]


def test_evaluate_report_requires_with_exception_approval() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={})
    packages = [
        module.PackageLicense("unknown", "1", "Apache-2.0 WITH Totally-Unknown-terms", ""),
        module.PackageLicense("rejected", "1", "GPL-2.0-only WITH Classpath-exception-2.0", ""),
    ]

    violations, _ = module.evaluate_report(packages, policy)

    assert violations == [
        "rejected==1: disallowed license(s): GPL-2.0-only WITH Classpath-exception-2.0",
        "unknown==1: disallowed license(s): Apache-2.0 WITH Totally-Unknown-terms",
    ]

    approved_policy = module.LicensePolicy(
        allowed_licenses=frozenset({"Apache-2.0", "Apache-2.0 WITH LLVM-exception"}), exceptions={}
    )
    approved = module.PackageLicense("approved", "1", "Apache-2.0 WITH LLVM-exception", "")

    approved_violations, _ = module.evaluate_report([approved], approved_policy)

    assert approved_violations == []


def test_evaluate_report_ignores_empty_semicolon_segments() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0", "MIT"}), exceptions={})
    package = module.PackageLicense("trailing", "1", "Apache Software License;; ", "")

    violations, _ = module.evaluate_report([package], policy)

    assert violations == []


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
    packages = [
        module.PackageLicense(
            "missing-metadata",
            "1",
            "UNKNOWN",
            MIT_LICENSE_TEXT,
        ),
        module.PackageLicense(
            "alternate-heading",
            "1",
            "UNKNOWN",
            MIT_LICENSE_TEXT.replace("MIT License", "The MIT License (MIT)", 1),
        ),
    ]

    violations, _ = module.evaluate_report(packages, policy)

    assert violations == []


def test_evaluate_report_rejects_incomplete_or_modified_mit_license_text() -> None:
    module = load_script()
    policy = module.LicensePolicy(allowed_licenses=frozenset({"MIT"}), exceptions={})
    packages = [
        module.PackageLicense(
            "additional-restriction",
            "1",
            "UNKNOWN",
            f"{MIT_LICENSE_TEXT}\nUse is prohibited in commercial products.",
        ),
        module.PackageLicense(
            "truncated",
            "1",
            "UNKNOWN",
            "MIT License\n\nCopyright example\n\nPermission is hereby granted, free of charge",
        ),
    ]

    violations, _ = module.evaluate_report(packages, policy)

    assert violations == [
        "additional-restriction==1: unknown or unrecognized license 'UNKNOWN'",
        "truncated==1: unknown or unrecognized license 'UNKNOWN'",
    ]


def test_evaluate_report_requires_exception_license_to_remain_unchanged() -> None:
    module = load_script()
    exception = module.ExceptionPolicy(
        reports=frozenset({module.ExceptionReport(version="1", license="MPL-2.0")}),
        reason="Reviewed separately.",
    )
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={"example": exception})
    package = module.PackageLicense("example", "2", "GPL-3.0-only", "")

    violations, reviewed = module.evaluate_report([package], policy)

    assert violations == ["example==2: exception does not approve 'GPL-3.0-only'; expected one of: 1 with 'MPL-2.0'"]
    assert reviewed == []


def test_evaluate_report_accepts_each_exact_exception_report() -> None:
    module = load_script()
    exception = module.ExceptionPolicy(
        reports=frozenset(
            {
                module.ExceptionReport(version="1", license="Legacy license"),
                module.ExceptionReport(version="2", license="Current license"),
            }
        ),
        reason="Reviewed per locked slice.",
    )
    policy = module.LicensePolicy(allowed_licenses=frozenset(), exceptions={"example": exception})
    packages = [
        module.PackageLicense("example", "1", "Legacy license", ""),
        module.PackageLicense("example", "2", "Current license", ""),
    ]

    violations, reviewed = module.evaluate_report(packages, policy)

    assert violations == []
    assert reviewed == [
        "example==1: Legacy license (Reviewed per locked slice.)",
        "example==2: Current license (Reviewed per locked slice.)",
    ]


def test_evaluate_report_rejects_stale_exceptions() -> None:
    module = load_script()
    exception = module.ExceptionPolicy(
        reports=frozenset({module.ExceptionReport(version="1", license="MPL-2.0")}),
        reason="Reviewed separately.",
    )
    policy = module.LicensePolicy(allowed_licenses=frozenset({"Apache-2.0"}), exceptions={"removed": exception})

    violations, _ = module.evaluate_report([], policy)

    assert violations == ["removed: stale exception; package is not present in the scanned environment"]
