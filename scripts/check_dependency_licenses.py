# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PIP_LICENSES_VERSION = "5.5.5"
POLICY_PATH = Path(__file__).resolve().parents[1] / "dependency-license-policy.toml"

LICENSE_ALIASES = {
    "Apache Software License": {"Apache-2.0"},
    "BSD License": {"BSD-3-Clause"},
    "ISC License (ISCL)": {"ISC"},
    "MIT License": {"MIT"},
    "Python Software Foundation License": {"PSF-2.0"},
    "The Unlicense (Unlicense)": {"Unlicense"},
}
EXPRESSION_OPERATOR = re.compile(r"\s+(?:AND|OR|WITH)\s+|[()]", re.IGNORECASE)


@dataclass(frozen=True)
class ExceptionPolicy:
    license: str
    reason: str


@dataclass(frozen=True)
class LicensePolicy:
    allowed_licenses: frozenset[str]
    exceptions: dict[str, ExceptionPolicy]


@dataclass(frozen=True)
class PackageLicense:
    name: str
    version: str
    license: str
    license_text: str


def load_policy(path: Path = POLICY_PATH) -> LicensePolicy:
    """Load the dependency-license policy from TOML."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    exceptions = {
        name.casefold(): ExceptionPolicy(license=value["license"], reason=value["reason"])
        for name, value in data.get("exceptions", {}).items()
    }
    return LicensePolicy(allowed_licenses=frozenset(data["allowed_licenses"]), exceptions=exceptions)


def looks_like_mit_license(license_text: str) -> bool:
    normalized = " ".join(license_text.split()).casefold()
    return normalized.startswith("mit license ") and "permission is hereby granted, free of charge" in normalized


def normalized_license_ids(package: PackageLicense) -> frozenset[str]:
    """Normalize pip-licenses output into SPDX-like license identifiers."""
    reported = package.license.strip()
    if reported.casefold() == "unknown":
        return frozenset({"MIT"}) if looks_like_mit_license(package.license_text) else frozenset()

    if reported.startswith("MIT License\n") and looks_like_mit_license(reported):
        return frozenset({"MIT"})

    aliases = LICENSE_ALIASES.get(reported)
    if aliases is not None:
        return frozenset(aliases)

    identifiers: set[str] = set()
    for part in reported.split(";"):
        stripped = part.strip()
        part_aliases = LICENSE_ALIASES.get(stripped)
        if part_aliases is not None:
            identifiers.update(part_aliases)
            continue

        tokens = [token.strip() for token in EXPRESSION_OPERATOR.split(stripped) if token.strip()]
        if not tokens:
            return frozenset()
        identifiers.update(tokens)

    return frozenset(identifiers)


def parse_report(data: list[dict[str, Any]]) -> list[PackageLicense]:
    """Parse the structured pip-licenses report."""
    return [
        PackageLicense(
            name=str(item["Name"]),
            version=str(item["Version"]),
            license=str(item["License"]),
            license_text=str(item.get("LicenseText", "")),
        )
        for item in data
    ]


def evaluate_report(packages: list[PackageLicense], policy: LicensePolicy) -> tuple[list[str], list[str]]:
    """Return policy violations and reviewed-exception descriptions."""
    violations: list[str] = []
    reviewed: list[str] = []
    seen_exceptions: set[str] = set()

    for package in sorted(packages, key=lambda item: item.name.casefold()):
        package_key = package.name.casefold()
        exception = policy.exceptions.get(package_key)
        if exception is not None:
            seen_exceptions.add(package_key)
            if package.license != exception.license:
                violations.append(
                    f"{package.name}=={package.version}: exception expected {exception.license!r}, "
                    f"but package reports {package.license!r}"
                )
            else:
                reviewed.append(f"{package.name}=={package.version}: {package.license} ({exception.reason})")
            continue

        identifiers = normalized_license_ids(package)
        if not identifiers:
            violations.append(f"{package.name}=={package.version}: unknown or unrecognized license {package.license!r}")
        elif not identifiers.issubset(policy.allowed_licenses):
            disallowed = ", ".join(sorted(identifiers - policy.allowed_licenses))
            violations.append(f"{package.name}=={package.version}: disallowed license(s): {disallowed}")

    for package_key in sorted(policy.exceptions.keys() - seen_exceptions):
        violations.append(f"{package_key}: stale exception; package is not present in the scanned environment")

    return violations, reviewed


def collect_report() -> list[PackageLicense]:
    """Run the pinned scanner against the active Python environment."""
    command = [
        "uv",
        "tool",
        "run",
        "--from",
        f"pip-licenses=={PIP_LICENSES_VERSION}",
        "pip-licenses",
        "--python",
        sys.executable,
        "--from=mixed",
        "--format=json",
        "--with-license-file",
        "--no-license-path",
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"pip-licenses exited with status {result.returncode}")
    return parse_report(json.loads(result.stdout))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime dependencies against the Apache-2.0 license policy")
    parser.add_argument("--report", type=Path, help="Read a pip-licenses JSON report instead of running the scanner")
    args = parser.parse_args()

    if args.report is None:
        packages = collect_report()
    else:
        packages = parse_report(json.loads(args.report.read_text(encoding="utf-8")))

    violations, reviewed = evaluate_report(packages, load_policy())
    print(f"Checked {len(packages)} installed runtime packages.")

    if reviewed:
        print("\nReviewed package-specific exceptions:")
        for item in reviewed:
            print(f"  - {item}")

    if violations:
        print("\nDependency license policy violations:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print("\nAll dependency licenses satisfy the Apache-2.0 compatibility policy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
