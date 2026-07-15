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
EXPRESSION_TOKEN = re.compile(r"(\(|\)|\bAND\b|\bOR\b|\bWITH\b)")


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
    has_mit_heading = normalized.startswith(("mit license ", "the mit license (mit) "))
    return has_mit_heading and "permission is hereby granted, free of charge" in normalized


def _and_alternatives(
    left: tuple[frozenset[str], ...], right: tuple[frozenset[str], ...]
) -> tuple[frozenset[str], ...]:
    return tuple(left_ids | right_ids for left_ids in left for right_ids in right)


class LicenseExpressionParser:
    """Parse the SPDX operators needed by package license metadata."""

    def __init__(self, expression: str) -> None:
        self.tokens = [token.strip() for token in EXPRESSION_TOKEN.split(expression) if token.strip()]
        self.position = 0

    def parse(self) -> tuple[frozenset[str], ...]:
        alternatives = self._parse_or()
        if self.position != len(self.tokens):
            return ()
        return alternatives

    def _parse_or(self) -> tuple[frozenset[str], ...]:
        alternatives = self._parse_and()
        while self._accept("OR"):
            right = self._parse_and()
            if not right:
                return ()
            alternatives += right
        return alternatives

    def _parse_and(self) -> tuple[frozenset[str], ...]:
        alternatives = self._parse_primary()
        while self._accept("AND"):
            alternatives = _and_alternatives(alternatives, self._parse_primary())
        return alternatives

    def _parse_primary(self) -> tuple[frozenset[str], ...]:
        if self._accept("("):
            alternatives = self._parse_or()
            if not self._accept(")"):
                return ()
            return alternatives

        if self.position >= len(self.tokens) or self.tokens[self.position] in {"AND", "OR", "WITH", ")"}:
            return ()

        reported_id = self.tokens[self.position]
        self.position += 1
        identifiers = frozenset(LICENSE_ALIASES.get(reported_id, {reported_id}))

        if self._accept("WITH"):
            if self.position >= len(self.tokens) or self.tokens[self.position] in {"(", ")", "AND", "OR", "WITH"}:
                return ()
            self.position += 1

        return (identifiers,)

    def _accept(self, token: str) -> bool:
        if self.position >= len(self.tokens) or self.tokens[self.position] != token:
            return False
        self.position += 1
        return True


def normalized_license_alternatives(package: PackageLicense) -> tuple[frozenset[str], ...]:
    """Normalize a reported license into alternatives of required SPDX-like identifiers."""
    reported = package.license.strip()
    if reported.casefold() == "unknown":
        return (frozenset({"MIT"}),) if looks_like_mit_license(package.license_text) else ()

    if reported.startswith("MIT License\n") and looks_like_mit_license(reported):
        return (frozenset({"MIT"}),)

    aliases = LICENSE_ALIASES.get(reported)
    if aliases is not None:
        return (frozenset(aliases),)

    alternatives: tuple[frozenset[str], ...] = (frozenset(),)
    parsed_any = False
    for part in reported.split(";"):
        stripped = part.strip()
        if not stripped:
            continue

        parsed = LicenseExpressionParser(stripped).parse()
        if not parsed:
            return ()
        alternatives = _and_alternatives(alternatives, parsed)
        parsed_any = True

    return alternatives if parsed_any else ()


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

        alternatives = normalized_license_alternatives(package)
        if not alternatives:
            violations.append(f"{package.name}=={package.version}: unknown or unrecognized license {package.license!r}")
        elif not any(identifiers.issubset(policy.allowed_licenses) for identifiers in alternatives):
            disallowed_ids = set().union(*(identifiers - policy.allowed_licenses for identifiers in alternatives))
            disallowed = ", ".join(sorted(disallowed_ids))
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
