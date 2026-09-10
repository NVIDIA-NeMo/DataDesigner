# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[3]
AUDIT_SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_slurm_public_artifacts.py"
PACKAGE_LICENSE = REPOSITORY_ROOT / "packages" / "data-designer-slurm" / "LICENSE"
SPDX_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""


def test_repository_slurm_artifacts_pass_public_audit() -> None:
    result = _run_audit()

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("content", "rule"),
    (
        (f"token=github_pat_{'a' * 24}\n", "GitHub token"),
        ("workspace=/home/specific-user/run\n", "user-specific absolute path"),
        ("endpoint=10.23.45.67\n", "private infrastructure address"),
        ("host=service.internal.nvidia.com\n", "internal NVIDIA hostname"),
    ),
    ids=("credential", "user-path", "private-address", "internal-host"),
)
def test_public_audit_reports_rule_without_echoing_sensitive_content(
    tmp_path: Path,
    content: str,
    rule: str,
) -> None:
    artifact = tmp_path / "runtime.log"
    artifact.write_text(content)

    result = _run_audit(artifact)

    assert result.returncode == 1
    assert rule in result.stderr
    assert content.strip() not in result.stderr
    assert str(tmp_path) not in result.stderr


def test_public_audit_checks_wheel_members_and_license_text(tmp_path: Path) -> None:
    wheel = tmp_path / "data_designer_slurm-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("data_designer/slurm/runtime.py", f"{SPDX_HEADER}\nfrom __future__ import annotations\n")
        archive.writestr(
            "data_designer_slurm-1.0.0.dist-info/licenses/LICENSE",
            PACKAGE_LICENSE.read_bytes(),
        )
        archive.writestr(
            "data_designer_slurm-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.5\nLicense-Expression: Apache-2.0\n",
        )

    result = _run_audit(wheel)

    assert result.returncode == 0, result.stderr


def test_public_audit_rejects_wheel_with_truncated_license(tmp_path: Path) -> None:
    wheel = tmp_path / "data_designer_slurm-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr(
            "data_designer_slurm-1.0.0.dist-info/licenses/LICENSE",
            "Apache License\nVersion 2.0\n",
        )
        archive.writestr(
            "data_designer_slurm-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.5\nLicense-Expression: Apache-2.0\n",
        )

    result = _run_audit(wheel)

    assert result.returncode == 1
    assert "wheel does not contain exactly one canonical license text" in result.stderr


def test_public_audit_rejects_multiple_wheel_distribution_roots(tmp_path: Path) -> None:
    wheel = tmp_path / "data_designer_slurm-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr(
            "data_designer_slurm-1.0.0.dist-info/licenses/LICENSE",
            PACKAGE_LICENSE.read_bytes(),
        )
        archive.writestr(
            "decoy-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.5\nLicense-Expression: Apache-2.0\n",
        )

    result = _run_audit(wheel)

    assert result.returncode == 1
    assert "wheel must contain exactly one .dist-info root" in result.stderr


@pytest.mark.parametrize(
    ("extra_member", "expected"),
    (
        (
            "data_designer_slurm-1.0.0.dist-info/licenses/LICENSE.txt",
            "wheel does not contain exactly one canonical license text",
        ),
        (
            "data_designer_slurm-1.0.0.dist-info/metadata",
            "wheel does not contain exactly one Apache-2.0 metadata record",
        ),
    ),
    ids=("license", "metadata"),
)
def test_public_audit_rejects_duplicate_wheel_distribution_records(
    tmp_path: Path,
    extra_member: str,
    expected: str,
) -> None:
    wheel = tmp_path / "data_designer_slurm-1.0.0-py3-none-any.whl"
    metadata = "Metadata-Version: 2.5\nLicense-Expression: Apache-2.0\n"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("data_designer_slurm-1.0.0.dist-info/licenses/LICENSE", PACKAGE_LICENSE.read_bytes())
        archive.writestr("data_designer_slurm-1.0.0.dist-info/METADATA", metadata)
        archive.writestr(extra_member, metadata if extra_member.endswith("metadata") else PACKAGE_LICENSE.read_bytes())

    result = _run_audit(wheel)

    assert result.returncode == 1
    assert expected in result.stderr


def test_public_audit_rejects_unsafe_or_unlicensed_wheel_members(tmp_path: Path) -> None:
    wheel = tmp_path / "data_designer_slurm-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("../escaped.py", "pass\n")
        archive.writestr("data_designer/slurm/runtime.py", "from __future__ import annotations\n")

    result = _run_audit(wheel)

    assert result.returncode == 1
    assert "archive member path is unsafe" in result.stderr
    assert "missing the NVIDIA Apache-2.0 SPDX header" in result.stderr
    assert "wheel must contain exactly one .dist-info root" in result.stderr


def test_public_audit_scans_archive_member_names_without_echoing_them(tmp_path: Path) -> None:
    wheel = tmp_path / "data_designer_slurm-1.0.0-py3-none-any.whl"
    sensitive_member = "data_designer/slurm/10.23.45.67.py"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr(sensitive_member, SPDX_HEADER)
        archive.writestr(
            "data_designer_slurm-1.0.0.dist-info/licenses/LICENSE",
            PACKAGE_LICENSE.read_bytes(),
        )
        archive.writestr(
            "data_designer_slurm-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.5\nLicense-Expression: Apache-2.0\n",
        )

    result = _run_audit(wheel)

    assert result.returncode == 1
    assert "private infrastructure address" in result.stderr
    assert sensitive_member not in result.stderr


def test_public_audit_scans_unknown_suffix_archive_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("credentials.env", f"token=github_pat_{'a' * 24}\n")

    result = _run_audit(archive_path)

    assert result.returncode == 1
    assert "GitHub token" in result.stderr


def test_public_audit_detects_zip_archive_by_content(tmp_path: Path) -> None:
    archive_path = tmp_path / "opaque-artifact"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("credentials.env", f"token=github_pat_{'a' * 24}\n")

    result = _run_audit(archive_path)

    assert result.returncode == 1
    assert "GitHub token" in result.stderr


def test_public_audit_detects_tar_archive_by_content(tmp_path: Path) -> None:
    archive_path = tmp_path / "opaque-artifact"
    secret = f"nvapi-{'a' * 24}".encode()
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("credentials.env")
        member.size = len(secret)
        archive.addfile(member, BytesIO(secret))

    result = _run_audit(archive_path)

    assert result.returncode == 1
    assert "NGC API key" in result.stderr


def test_public_audit_checks_runtime_tar_content_and_entrypoint_license(tmp_path: Path) -> None:
    archive_path = tmp_path / "runtime.tar.gz"
    secret = f"nvapi-{'a' * 24}"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        entrypoint = tarfile.TarInfo("entrypoint.sh")
        entrypoint_content = f"#!/bin/sh\necho {secret}\n".encode()
        entrypoint.size = len(entrypoint_content)
        archive.addfile(entrypoint, BytesIO(entrypoint_content))

    result = _run_audit(archive_path)

    assert result.returncode == 1
    assert "NGC API key" in result.stderr
    assert "missing the NVIDIA Apache-2.0 SPDX header" in result.stderr
    assert secret not in result.stderr


@pytest.mark.parametrize(
    "member_type",
    (tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE),
    ids=("fifo", "character-device", "block-device"),
)
def test_public_audit_rejects_tar_special_members(tmp_path: Path, member_type: bytes) -> None:
    archive_path = tmp_path / "runtime.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        member = tarfile.TarInfo("unsupported-member")
        member.type = member_type
        archive.addfile(member)

    result = _run_audit(archive_path)

    assert result.returncode == 1
    assert "archive member type is unsupported" in result.stderr


def test_public_audit_allows_tar_directories(tmp_path: Path) -> None:
    archive_path = tmp_path / "runtime.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        directory = tarfile.TarInfo("runtime")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)

    result = _run_audit(archive_path)

    assert result.returncode == 0, result.stderr


def test_public_audit_rejects_explicit_symbolic_link_without_disclosing_its_parent(tmp_path: Path) -> None:
    target = tmp_path / "target.log"
    target.write_text("safe\n")
    link = tmp_path / "linked.log"
    link.symlink_to(target)

    result = _run_audit(link)

    assert result.returncode == 1
    assert "symbolic-link artifact requires explicit review" in result.stderr
    assert str(tmp_path) not in result.stderr


def test_public_audit_rejects_symbolic_link_named_as_generated_cache(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (artifacts / "__pycache__").symlink_to(target, target_is_directory=True)

    result = _run_audit(artifacts)

    assert result.returncode == 1
    assert "symbolic-link artifact requires explicit review" in result.stderr


@pytest.mark.parametrize("filename", ("credentials.env", "records.csv", "METADATA", "LICENSE"))
def test_public_audit_scans_content_with_unknown_or_empty_suffixes(tmp_path: Path, filename: str) -> None:
    artifact = tmp_path / filename
    artifact.write_text(f"token=github_pat_{'a' * 24}\n")

    result = _run_audit(artifact)

    assert result.returncode == 1
    assert "GitHub token" in result.stderr


def test_public_audit_does_not_allow_test_sentinels_outside_exact_sources(tmp_path: Path) -> None:
    artifact = tmp_path / "test_credentials.py"
    artifact.write_text('secret = "super-secret-token"\n')

    result = _run_audit(artifact)

    assert result.returncode == 1
    assert "plaintext secret assignment" in result.stderr


def _run_audit(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), *(str(path) for path in paths)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
