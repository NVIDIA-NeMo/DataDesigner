# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from packaging.tags import interpreter_name, interpreter_version
from packaging.utils import canonicalize_name, parse_wheel_filename

from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.filesystem import compute_file_sha256, ensure_private_directory, read_regular_bytes
from data_designer.slurm.client.records import ClientErrorCode, ClientInstallerOutcome
from data_designer.slurm.contracts import ArtifactReference, InstalledDistribution

DistributionInventory = Callable[[Path | None], tuple[InstalledDistribution, ...]]
CommandRunner = Callable[[tuple[str, ...]], None]


@dataclass(frozen=True)
class PreparedClientEnvironment:
    run_id: str
    shard_id: str
    attempt_id: str
    attempt_dir: Path
    overlay_path: Path
    dependency_lock: ArtifactReference
    client_image_sha256: str
    python_abi: str
    installer_outcome: ClientInstallerOutcome
    installed_distributions: tuple[InstalledDistribution, ...]


@dataclass(frozen=True)
class _BootstrapInputs:
    run_id: str
    run_root: Path
    shard_id: str
    attempt_id: str
    attempt_dir: Path
    client_image_sha256: str
    python_abi: str
    installer_path: Path
    inspection: dict[str, object]
    dependency_lock: ArtifactReference


@dataclass(frozen=True)
class _VerifiedDependencies:
    image_distributions: tuple[InstalledDistribution, ...]
    overlay_packages: tuple[dict[str, object], ...]


class ClientEnvironmentBuilder:
    """Prepare one immutable attempt-local package environment."""

    def __init__(
        self,
        *,
        inventory: DistributionInventory | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._inventory = inventory or inspect_distributions
        self._command_runner = command_runner or _run_installer

    def prepare(
        self,
        plan_path: Path,
        *,
        shard_id: str,
        attempt_id: str,
        attempt_dir: Path,
    ) -> PreparedClientEnvironment:
        """Verify the plan and lock subset needed before plugin-aware imports."""
        inputs = self._load_bootstrap_inputs(
            plan_path,
            shard_id=shard_id,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
        )
        dependencies = self._verify_dependency_lock(inputs)
        overlay_path, installer_outcome, installed = self._prepare_overlay(inputs, dependencies)
        return PreparedClientEnvironment(
            run_id=inputs.run_id,
            shard_id=inputs.shard_id,
            attempt_id=inputs.attempt_id,
            attempt_dir=inputs.attempt_dir,
            overlay_path=overlay_path,
            dependency_lock=inputs.dependency_lock,
            client_image_sha256=inputs.client_image_sha256,
            python_abi=inputs.python_abi,
            installer_outcome=installer_outcome,
            installed_distributions=installed,
        )

    @staticmethod
    def _load_bootstrap_inputs(
        plan_path: Path,
        *,
        shard_id: str,
        attempt_id: str,
        attempt_dir: Path,
    ) -> _BootstrapInputs:
        _validate_input_path(plan_path, "resolved plan")
        _validate_input_path(attempt_dir, "attempt directory")
        if not re.fullmatch(r"shard-[0-9]{5,}", shard_id):
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "shard identifier is invalid")
        if not re.fullmatch(r"attempt-[0-9]{4,}", attempt_id) or int(attempt_id.removeprefix("attempt-")) < 1:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "attempt identifier is invalid")
        plan = _load_json_object(plan_path, ClientErrorCode.INVALID_INPUT)
        run_id = _require_string(plan, "run_id")
        run_root = plan_path.parent
        if plan_path.name != "resolved-plan.json" or run_root.name != run_id or run_root.parent.name != "runs":
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "resolved plan path is not canonical")
        expected_attempt = run_root / "shards" / shard_id / "attempts" / attempt_id
        if attempt_dir.as_posix() != expected_attempt.as_posix():
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "attempt directory does not match the plan")
        ensure_private_directory(attempt_dir)

        client = _require_object(plan, "client")
        image = _require_object(client, "image")
        image_sha256 = _require_digest(image, "sha256")
        inspection_record = _require_object(image, "inspection")
        inspection = _require_object(inspection_record, "inspection")
        if inspection.get("kind") != "client":
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "resolved client image inspection is invalid")
        python_abi = _require_string(inspection, "python_abi")
        if python_abi != f"{interpreter_name()}{interpreter_version()}":
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "client Python ABI differs from the plan")
        installer_path = Path(_require_string(inspection, "installer_path"))
        if not installer_path.is_absolute() or installer_path.as_posix() == "/":
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client installer path is invalid")

        lock_reference = _artifact_reference(_require_object(client, "dependency_lock"))
        if Path(lock_reference.path).as_posix() != (run_root / "dependency-lock.json").as_posix():
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency lock path is not canonical")
        return _BootstrapInputs(
            run_id=run_id,
            run_root=run_root,
            shard_id=shard_id,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            client_image_sha256=image_sha256,
            python_abi=python_abi,
            installer_path=installer_path,
            inspection=inspection,
            dependency_lock=lock_reference,
        )

    def _verify_dependency_lock(self, inputs: _BootstrapInputs) -> _VerifiedDependencies:
        lock_bytes = read_regular_bytes(
            Path(inputs.dependency_lock.path), missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING
        )
        if _sha256_bytes(lock_bytes) != inputs.dependency_lock.sha256:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_DIGEST_MISMATCH, "dependency lock digest differs")
        lock = _parse_json_bytes(lock_bytes, ClientErrorCode.DEPENDENCY_CONFLICT)
        if _require_digest(lock, "client_image_sha256") != inputs.client_image_sha256:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency lock targets another image")
        if _require_string(lock, "python_abi") != inputs.python_abi:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency lock targets another Python ABI")

        expected_image = _parse_distributions(lock.get("image_distributions"))
        inspected_image = _parse_distributions(inputs.inspection.get("distributions"))
        try:
            actual_image = self._inventory(None)
        except ClientWorkerError:
            raise
        except Exception as error:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_CONFLICT, "client image inventory cannot be verified"
            ) from error
        if expected_image != inspected_image or actual_image != expected_image:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "client image inventory differs from the lock")

        source = lock.get("source")
        if source is not None:
            source_reference = _artifact_reference(_as_object(source))
            _verify_input_artifact(source_reference, inputs.run_root / "inputs")
        return _VerifiedDependencies(
            image_distributions=expected_image,
            overlay_packages=tuple(_as_object_list(lock.get("overlay_packages"))),
        )

    def _prepare_overlay(
        self,
        inputs: _BootstrapInputs,
        dependencies: _VerifiedDependencies,
    ) -> tuple[Path, ClientInstallerOutcome, tuple[InstalledDistribution, ...]]:
        expected_overlay, wheels = _verify_wheels(
            dependencies.overlay_packages,
            inputs.run_root / "dependencies",
            dependencies.image_distributions,
        )
        overlay_path = inputs.attempt_dir / "client-env" / "site-packages"
        outcome = self._install_overlay(inputs.installer_path, wheels, expected_overlay, overlay_path)
        installed = tuple(sorted((*dependencies.image_distributions, *expected_overlay), key=lambda item: item.name))
        return overlay_path, outcome, installed

    def _install_overlay(
        self,
        installer_path: Path,
        wheels: tuple[Path, ...],
        expected: tuple[InstalledDistribution, ...],
        target: Path,
    ) -> ClientInstallerOutcome:
        ensure_private_directory(target)
        existing = self._inventory(target)
        if existing == expected and any(target.iterdir()):
            return ClientInstallerOutcome.REUSED
        if any(target.iterdir()):
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "attempt overlay contains unexpected files")
        if not wheels:
            return ClientInstallerOutcome.NOT_REQUIRED
        try:
            self._command_runner(
                (
                    installer_path.as_posix(),
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--no-index",
                    "--target",
                    target.as_posix(),
                    *(wheel.as_posix() for wheel in wheels),
                )
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_INSTALL_FAILED, "client dependency installation failed"
            ) from error
        if self._inventory(target) != expected:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_INSTALL_FAILED, "installed client dependencies differ from the lock"
            )
        return ClientInstallerOutcome.INSTALLED


def inspect_distributions(path: Path | None) -> tuple[InstalledDistribution, ...]:
    """Return an exact immutable distribution inventory."""
    distributions = (
        importlib.metadata.distributions() if path is None else importlib.metadata.distributions(path=[path.as_posix()])
    )
    installed: list[InstalledDistribution] = []
    names: set[str] = set()
    for distribution in distributions:
        name = canonicalize_name(distribution.metadata["Name"] or "")
        version = distribution.version
        if not name or not version or name in names:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "installed distribution inventory is invalid")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url is not None and _is_mutable_direct_url(direct_url):
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_CONFLICT, "mutable installed distributions are forbidden"
            )
        names.add(name)
        installed.append(InstalledDistribution(name=name, version=version))
    return tuple(sorted(installed, key=lambda item: item.name))


def activate_environment(prepared: PreparedClientEnvironment) -> None:
    """Activate one verified overlay and isolate Data Designer attempt state."""
    home = prepared.attempt_dir / "data-designer-home"
    cache = prepared.attempt_dir / "cache"
    ensure_private_directory(home)
    ensure_private_directory(cache)
    os.environ["DATA_DESIGNER_HOME"] = home.as_posix()
    os.environ["XDG_CACHE_HOME"] = cache.as_posix()
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["DISABLE_DATA_DESIGNER_PLUGINS"] = "false"
    if prepared.overlay_path.as_posix() not in sys.path:
        sys.path.append(prepared.overlay_path.as_posix())
    importlib.invalidate_caches()


def _validate_input_path(path: Path, label: str) -> None:
    if not path.is_absolute() or path != Path(os.path.normpath(path.as_posix())):
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, f"{label} path is not canonical")


def _run_installer(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _verify_wheels(
    packages: tuple[dict[str, object], ...],
    dependencies_root: Path,
    image_distributions: tuple[InstalledDistribution, ...],
) -> tuple[tuple[InstalledDistribution, ...], tuple[Path, ...]]:
    expected: list[InstalledDistribution] = []
    wheels: list[Path] = []
    image_names = {item.name for item in image_distributions}
    for package in packages:
        name = canonicalize_name(_require_string(package, "name"))
        version = _require_string(package, "version")
        if name in image_names or name in {item.name for item in expected}:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency distributions overlap")
        artifact = _artifact_reference(_require_object(package, "artifact"))
        wheel = Path(artifact.path)
        try:
            wheel.relative_to(dependencies_root)
        except ValueError as error:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency wheel path is not canonical") from error
        if wheel.parent != dependencies_root or wheel.suffix != ".whl":
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency wheel path is not canonical")
        if compute_file_sha256(wheel, missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING) != artifact.sha256:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_DIGEST_MISMATCH, "dependency wheel digest differs")
        try:
            wheel_name, wheel_version, _, _ = parse_wheel_filename(wheel.name)
        except ValueError as error:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_CONFLICT, "dependency artifact is not a valid wheel"
            ) from error
        if canonicalize_name(wheel_name) != name or str(wheel_version) != version:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_CONFLICT, "dependency wheel identity differs from the lock"
            )
        expected.append(InstalledDistribution(name=name, version=version))
        wheels.append(wheel)
    sorted_pairs = sorted(zip(expected, wheels, strict=True), key=lambda pair: pair[0].name)
    if tuple(item.name for item in expected) != tuple(pair[0].name for pair in sorted_pairs):
        raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency overlay is not sorted")
    return tuple(pair[0] for pair in sorted_pairs), tuple(pair[1] for pair in sorted_pairs)


def _verify_input_artifact(reference: ArtifactReference, root: Path) -> None:
    path = Path(reference.path)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency source path is not canonical") from error
    if path.parent != root or path.suffix != ".json":
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency source path is not canonical")
    if compute_file_sha256(path, missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING) != reference.sha256:
        raise ClientWorkerError(ClientErrorCode.DEPENDENCY_DIGEST_MISMATCH, "dependency source digest differs")


def _parse_distributions(value: object) -> tuple[InstalledDistribution, ...]:
    parsed = tuple(
        InstalledDistribution(
            name=canonicalize_name(_require_string(item, "name")), version=_require_string(item, "version")
        )
        for item in _as_object_list(value)
    )
    names = tuple(item.name for item in parsed)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency inventory is not sorted and unique")
    return parsed


def _load_json_object(path: Path, code: ClientErrorCode) -> dict[str, object]:
    return _parse_json_bytes(read_regular_bytes(path, missing_code=code), code)


def _parse_json_bytes(value: bytes, code: ClientErrorCode) -> dict[str, object]:
    try:
        return _as_object(json.loads(value))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ClientWorkerError(code, "client JSON artifact is invalid") from error


def _as_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client JSON object is invalid")
    return cast(dict[str, object], value)


def _as_object_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency package list is invalid")
    return [_as_object(item) for item in value]


def _require_object(value: dict[str, object], key: str) -> dict[str, object]:
    try:
        return _as_object(value[key])
    except KeyError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "required client plan field is missing") from error


def _require_string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item or any(ord(character) < 32 for character in item):
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "required client string field is invalid")
    return item


def _require_digest(value: dict[str, object], key: str) -> str:
    digest = _require_string(value, key)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "required client digest is invalid")
    return digest


def _artifact_reference(value: dict[str, object]) -> ArtifactReference:
    try:
        return ArtifactReference(path=_require_string(value, "path"), sha256=_require_digest(value, "sha256"))
    except ValueError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client artifact reference is invalid") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_mutable_direct_url(value: str) -> bool:
    try:
        direct_url = _as_object(json.loads(value))
    except (json.JSONDecodeError, TypeError):
        return True
    if "dir_info" in direct_url or "vcs_info" in direct_url:
        return True
    archive_info = direct_url.get("archive_info")
    if archive_info is None:
        return True
    if not isinstance(archive_info, dict):
        return True
    hashes = archive_info.get("hashes")
    return not isinstance(hashes, dict) or not isinstance(hashes.get("sha256"), str)
