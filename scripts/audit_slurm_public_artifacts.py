# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scan public Slurm artifacts without rendering matched sensitive content.

The default scope covers deployable source, test source and fixtures, package
metadata, and release scripts. A small path-scoped allowlist masks only exact
synthetic test sentinels. Generic example.test hosts, loopback addresses, and
/workspace paths are the only implicit test-data allowances.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

MAXIMUM_MEMBER_SIZE = 16 * 1024 * 1024
MAXIMUM_ARCHIVE_CONTENT_SIZE = 128 * 1024 * 1024
MAXIMUM_ARCHIVE_MEMBERS = 10_000
REPOSITORY_ROOT = Path(__file__).parents[1]
CANONICAL_PACKAGE_LICENSE = REPOSITORY_ROOT / "packages" / "data-designer-slurm" / "LICENSE"
DEFAULT_ARTIFACTS = (
    "packages/data-designer-slurm/src",
    "packages/data-designer-slurm/README.md",
    "packages/data-designer-slurm/pyproject.toml",
    "packages/data-designer-slurm/LICENSE",
    "packages/data-designer-slurm/tests",
    "plans/850/data-designer-contract.md",
    "plans/870/slurm-early-security-review.md",
    "scripts/publish.sh",
    "scripts/test_slurm_package_install.py",
)
_LICENSED_SOURCE_SUFFIXES = frozenset({".py", ".rc", ".sh"})
_SPDX_COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
_SPDX_LICENSE = "SPDX-License-Identifier: Apache-2.0"
_SLURM_DIST_INFO_PATTERN = re.compile(r"^data_designer_slurm-[^/]+\.dist-info$")
_TEST_SOURCE_SENTINELS = {
    "packages/data-designer-slurm/tests/config/test_loading_builder.py": (
        '"clusters": {secret: profile_catalog.clusters["primary"]}',
        "super-secret-token",
        "sk_live_ABC123XYZ",
    ),
    "packages/data-designer-slurm/tests/contracts/test_config_records.py": ("plaintext-secret",),
    "packages/data-designer-slurm/tests/launcher/test_client.py": (
        "Authorization: Bearer bearer-secret;suffix status=failed",
        "Authorization: Bearer bearer-secret",
    ),
    "packages/data-designer-slurm/tests/planning/test_compiler.py": ("super-secret-token",),
    "packages/data-designer-slurm/tests/test_public_artifacts.py": (
        "/home/specific-user/run",
        "10.23.45.67",
        "service.internal.nvidia.com",
        "super-secret-token",
    ),
}


@dataclass(frozen=True, slots=True)
class AuditRule:
    """One high-confidence public-artifact rejection rule."""

    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One path-scoped finding that intentionally omits matched content."""

    location: str
    rule: str


@dataclass(frozen=True, slots=True)
class _ZipMemberAudit:
    findings: tuple[AuditFinding, ...]
    dist_info_root: str | None = None
    is_license: bool = False
    license_matches: bool = False
    is_metadata: bool = False
    metadata_declares_license: bool = False


_CONTENT_RULES = (
    AuditRule("private key material", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    AuditRule("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    AuditRule("GitHub token", re.compile(r"\b(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}\b")),
    AuditRule("NGC API key", re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}\b")),
    AuditRule("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    AuditRule(
        "plaintext secret assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret|credential)"
            r"\b\s*[:=]\s*[\"']?(?!<|\$|\{|\[)[A-Za-z0-9+/_.:@-]{16,}"
        ),
    ),
    AuditRule("authorization credential", re.compile(r"(?i)\bauthorization\s*:\s*(?:basic|bearer)\s+\S{16,}")),
    AuditRule(
        "internal NVIDIA hostname",
        re.compile(r"(?i)\b(?:(?:[a-z0-9-]+\.)*(?:corp|internal)|gitlab-master|urm)\.nvidia\.com\b"),
    ),
    AuditRule("user-specific absolute path", re.compile(r"/(?:Users|home|users)/[A-Za-z0-9._-]+(?:/|\b)")),
    AuditRule("site-specific shared path", re.compile(r"/(?:lustre|gpfs|fsx|net)/[^\s\"']+", re.IGNORECASE)),
    AuditRule(
        "private infrastructure address",
        re.compile(r"\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2})\b"),
    ),
)


def audit_public_artifacts(paths: Iterable[Path]) -> tuple[AuditFinding, ...]:
    """Return sanitized findings from files, directories, wheels, and tar archives."""
    findings: list[AuditFinding] = []
    for path in sorted((Path(os.path.abspath(candidate)) for candidate in paths), key=str):
        if path.is_symlink():
            findings.append(AuditFinding(_display_path(path), "symbolic-link artifact requires explicit review"))
        elif not path.exists():
            findings.append(AuditFinding(_display_path(path), "artifact does not exist"))
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_symlink():
                    findings.append(
                        AuditFinding(_display_path(child), "symbolic-link artifact requires explicit review")
                    )
                elif "__pycache__" in child.parts:
                    continue
                elif child.is_file():
                    findings.extend(_audit_file(child))
        else:
            findings.extend(_audit_file(path))
    return tuple(findings)


def _audit_file(path: Path) -> list[AuditFinding]:
    if _is_zip_archive(path):
        return _audit_zip(path)
    if _is_tar_archive(path):
        return _audit_tar(path)
    try:
        with path.open("rb") as stream:
            content = _read_bounded(stream, expected_size=path.stat().st_size)
    except OSError:
        return [AuditFinding(_display_path(path), "artifact cannot be read")]
    except ValueError as error:
        return [AuditFinding(_display_path(path), str(error))]
    location = _display_path(path)
    return _audit_content(
        location,
        location,
        content,
        allowed_sentinels=_TEST_SOURCE_SENTINELS.get(location, ()),
    )


def _audit_zip(path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    member_audits: list[_ZipMemberAudit] = []
    try:
        canonical_license = CANONICAL_PACKAGE_LICENSE.read_bytes()
    except OSError:
        return [AuditFinding(_display_path(path), "canonical package license cannot be read")]
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            limit_finding = _get_archive_limit_finding(
                path,
                member_count=len(members),
                content_size=sum(max(member.file_size, 0) for member in members),
            )
            if limit_finding is not None:
                return [limit_finding]
            for index, member in enumerate(sorted(members, key=lambda item: item.filename), start=1):
                location = f"{_display_path(path)}!member-{index}"
                member_audit = _audit_zip_member(
                    archive,
                    member,
                    location,
                    canonical_license,
                )
                findings.extend(member_audit.findings)
                member_audits.append(member_audit)
    except (OSError, zipfile.BadZipFile):
        return [AuditFinding(_display_path(path), "artifact is not a readable ZIP archive")]
    if path.suffix.casefold() == ".whl":
        findings.extend(_audit_wheel_distribution(path, member_audits))
    return findings


def _audit_tar(path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    content_size = 0
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member_count, member in enumerate(archive, start=1):
                content_size += max(member.size, 0)
                limit_finding = _get_archive_limit_finding(
                    path,
                    member_count=member_count,
                    content_size=content_size,
                )
                if limit_finding is not None:
                    return [limit_finding]
                location = f"{_display_path(path)}!member-{member_count}"
                findings.extend(_audit_tar_member(archive, member, location))
    except (OSError, tarfile.TarError):
        return [AuditFinding(_display_path(path), "artifact is not a readable tar archive")]
    return findings


def _audit_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    location: str,
    canonical_license: bytes,
) -> _ZipMemberAudit:
    dist_info_root = _get_dist_info_root(member.filename)
    if _is_unsafe_archive_name(member.filename):
        return _ZipMemberAudit((AuditFinding(location, "archive member path is unsafe"),))
    if stat.S_ISLNK(member.external_attr >> 16):
        return _ZipMemberAudit((AuditFinding(location, "archive member is a symbolic link"),))
    if member.is_dir():
        return _ZipMemberAudit((), dist_info_root=dist_info_root)
    is_license = _is_distribution_license(member.filename, dist_info_root)
    is_metadata = _is_distribution_metadata(member.filename, dist_info_root)
    try:
        with archive.open(member) as stream:
            content = _read_bounded(stream, expected_size=member.file_size)
    except (OSError, ValueError) as error:
        return _ZipMemberAudit((AuditFinding(location, str(error)),), dist_info_root=dist_info_root)
    return _ZipMemberAudit(
        findings=tuple(_audit_content(location, member.filename, content)),
        dist_info_root=dist_info_root,
        is_license=is_license,
        license_matches=is_license and content == canonical_license,
        is_metadata=is_metadata,
        metadata_declares_license=is_metadata and b"\nLicense-Expression: Apache-2.0\n" in b"\n" + content,
    )


def _audit_wheel_distribution(path: Path, members: Iterable[_ZipMemberAudit]) -> list[AuditFinding]:
    member_audits = tuple(members)
    roots = frozenset(member.dist_info_root for member in member_audits if member.dist_info_root is not None)
    location = _display_path(path)
    if len(roots) != 1:
        return [AuditFinding(location, "wheel must contain exactly one .dist-info root")]
    root = next(iter(roots))
    if _SLURM_DIST_INFO_PATTERN.fullmatch(root.casefold()) is None:
        return [AuditFinding(location, "wheel .dist-info root does not identify data-designer-slurm")]
    licenses = tuple(member for member in member_audits if member.dist_info_root == root and member.is_license)
    metadata = tuple(member for member in member_audits if member.dist_info_root == root and member.is_metadata)
    findings: list[AuditFinding] = []
    if len(licenses) != 1 or not licenses[0].license_matches:
        findings.append(AuditFinding(location, "wheel does not contain exactly one canonical license text"))
    if len(metadata) != 1 or not metadata[0].metadata_declares_license:
        findings.append(AuditFinding(location, "wheel does not contain exactly one Apache-2.0 metadata record"))
    return findings


def _audit_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    location: str,
) -> list[AuditFinding]:
    if _is_unsafe_archive_name(member.name):
        return [AuditFinding(location, "archive member path is unsafe")]
    if member.issym() or member.islnk():
        return [AuditFinding(location, "archive member is a link")]
    if member.isdir():
        return []
    if not member.isfile():
        return [AuditFinding(location, "archive member type is unsupported")]
    stream = archive.extractfile(member)
    if stream is None:
        return [AuditFinding(location, "archive member cannot be read")]
    try:
        with stream:
            content = _read_bounded(stream, expected_size=member.size)
    except (OSError, ValueError) as error:
        return [AuditFinding(location, str(error))]
    return _audit_content(location, member.name, content)


def _get_archive_limit_finding(path: Path, *, member_count: int, content_size: int) -> AuditFinding | None:
    if member_count > MAXIMUM_ARCHIVE_MEMBERS:
        return AuditFinding(_display_path(path), "archive exceeds the member-count scan limit")
    if content_size > MAXIMUM_ARCHIVE_CONTENT_SIZE:
        return AuditFinding(_display_path(path), "archive exceeds the expanded-content scan limit")
    return None


def _audit_content(
    location: str,
    logical_name: str,
    content: bytes,
    *,
    allowed_sentinels: Iterable[str] = (),
) -> list[AuditFinding]:
    text = content.decode("utf-8", errors="replace")
    for sentinel in sorted(allowed_sentinels, key=len, reverse=True):
        text = text.replace(sentinel, "<allowed-test-sentinel>")
    findings = [
        AuditFinding(location, rule.name)
        for rule in _CONTENT_RULES
        if rule.pattern.search(logical_name) or rule.pattern.search(text)
    ]
    if _requires_spdx(logical_name) and not _has_spdx_header(text):
        findings.append(AuditFinding(location, "packaged source is missing the NVIDIA Apache-2.0 SPDX header"))
    return findings


def _read_bounded(stream: BinaryIO, *, expected_size: int) -> bytes:
    if expected_size > MAXIMUM_MEMBER_SIZE:
        raise ValueError("artifact exceeds the per-file scan limit")
    content = stream.read(MAXIMUM_MEMBER_SIZE + 1)
    if len(content) > MAXIMUM_MEMBER_SIZE:
        raise ValueError("artifact exceeds the per-file scan limit")
    return content


def _requires_spdx(logical_name: str) -> bool:
    normalized = logical_name.replace("\\", "/")
    suffix = PurePosixPath(normalized).suffix.casefold()
    return suffix in _LICENSED_SOURCE_SUFFIXES and (
        normalized == "entrypoint.sh"
        or normalized.startswith("data_designer/slurm/")
        or "/src/data_designer/slurm/" in normalized
    )


def _has_spdx_header(text: str) -> bool:
    header = "\n".join(text.splitlines()[:5])
    return _SPDX_COPYRIGHT in header and _SPDX_LICENSE in header


def _get_dist_info_root(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if not parts or not parts[0].casefold().endswith(".dist-info"):
        return None
    return parts[0]


def _is_distribution_license(name: str, root: str | None) -> bool:
    if root is None:
        return False
    parts = tuple(part.casefold() for part in PurePosixPath(name).parts)
    license_names = {"license", "license.md", "license.txt"}
    return (len(parts) == 2 and parts[1] in license_names) or (
        len(parts) == 3 and parts[1] == "licenses" and parts[2] in license_names
    )


def _is_distribution_metadata(name: str, root: str | None) -> bool:
    return root is not None and name.casefold() == f"{root.casefold()}/metadata"


def _is_unsafe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts or "\\" in name


def _is_zip_archive(path: Path) -> bool:
    if path.suffix.casefold() in {".whl", ".zip"}:
        return True
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def _is_tar_archive(path: Path) -> bool:
    normalized = path.name.casefold()
    if normalized.endswith((".tar", ".tar.gz", ".tgz")):
        return True
    try:
        return tarfile.is_tarfile(path)
    except OSError:
        return False


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return "<external-artifact>"


def main(arguments: list[str] | None = None) -> int:
    """Run the public-artifact audit and print only path-scoped rule names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to scan")
    parsed = parser.parse_args(arguments)
    paths = parsed.paths or [REPOSITORY_ROOT / relative for relative in DEFAULT_ARTIFACTS]
    findings = audit_public_artifacts(paths)
    if findings:
        for finding in findings:
            print(f"{finding.location}: {finding.rule}", file=sys.stderr)
        return 1
    print(f"Slurm public-artifact audit passed for {len(paths)} target(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
