# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import stat
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
from data_designer.slurm.contracts import ArtifactReference, InstalledDistribution, canonical_json

DistributionInventory = Callable[[Path | None], tuple[InstalledDistribution, ...]]
CommandRunner = Callable[[tuple[str, ...]], None]


@dataclass(frozen=True)
class PreparedClientEnvironment:
    run_id: str
    shard_id: str
    attempt_id: str
    attempt_dir: Path
    scratch_root: Path
    overlay_path: Path
    dependency_lock: ArtifactReference
    client_runtime_sha256: str
    python_abi: str
    installer_outcome: ClientInstallerOutcome
    installed_distributions: tuple[InstalledDistribution, ...]


@dataclass(frozen=True)
class _BootstrapInputs:
    run_id: str
    logical_run_root: Path
    shard_id: str
    attempt_id: str
    attempt_dir: Path
    scratch_root: Path
    client_runtime_sha256: str
    python_abi: str
    python_executable: Path
    runtime: dict[str, object]
    dependency_lock: ArtifactReference


@dataclass(frozen=True)
class _VerifiedDependencies:
    base_distributions: tuple[InstalledDistribution, ...]
    overlay_packages: tuple[dict[str, object], ...]


class ClientEnvironmentBuilder:
    """Prepare one immutable allocation-local package environment."""

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
        scratch_root: Path,
    ) -> PreparedClientEnvironment:
        """Verify the plan and lock subset needed before plugin-aware imports."""
        inputs = self._load_bootstrap_inputs(
            plan_path,
            shard_id=shard_id,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            scratch_root=scratch_root,
        )
        dependencies = self._verify_dependency_lock(inputs)
        overlay_path, installer_outcome, installed = self._prepare_overlay(inputs, dependencies)
        return PreparedClientEnvironment(
            run_id=inputs.run_id,
            shard_id=inputs.shard_id,
            attempt_id=inputs.attempt_id,
            attempt_dir=inputs.attempt_dir,
            scratch_root=inputs.scratch_root,
            overlay_path=overlay_path,
            dependency_lock=inputs.dependency_lock,
            client_runtime_sha256=inputs.client_runtime_sha256,
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
        scratch_root: Path,
    ) -> _BootstrapInputs:
        _validate_input_path(plan_path, "resolved plan")
        _validate_input_path(attempt_dir, "attempt directory")
        _validate_scratch_root(scratch_root, attempt_dir)
        if not re.fullmatch(r"shard-[0-9]{5,}", shard_id):
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "shard identifier is invalid")
        if not re.fullmatch(r"attempt-[0-9]{4,}", attempt_id) or int(attempt_id.removeprefix("attempt-")) < 1:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "attempt identifier is invalid")
        plan_payload = _load_json_object(plan_path, ClientErrorCode.INVALID_INPUT)
        run_id = _require_string(plan_payload, "run_id")
        authored_reference = _artifact_reference(_require_object(plan_payload, "authored_config"))
        logical_run_root = Path(authored_reference.path).parent
        if (
            plan_path.name != "resolved-plan.json"
            or logical_run_root.name != run_id
            or logical_run_root.parent.name != "runs"
            or (logical_run_root / plan_path.name).as_posix() != plan_path.as_posix()
        ):
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "resolved plan path is not canonical")
        logical_attempt = logical_run_root / "shards" / shard_id / "attempts" / attempt_id
        expected_attempt = logical_attempt
        if attempt_dir.as_posix() != expected_attempt.as_posix():
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "attempt directory does not match the plan")
        ensure_private_directory(attempt_dir)

        client = _require_object(plan_payload, "client")
        runtime = _require_object(client, "runtime")
        runtime_sha256 = _require_digest(runtime, "runtime_sha256")
        runtime_identity = dict(runtime)
        runtime_identity.pop("runtime_sha256")
        if hashlib.sha256(canonical_json(runtime_identity)).hexdigest() != runtime_sha256:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client runtime fingerprint is invalid")
        python_abi = _require_string(runtime, "python_abi")
        if python_abi != f"{interpreter_name()}{interpreter_version()}":
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "client Python ABI differs from the plan")
        python_executable = Path(_require_string(runtime, "python_executable"))
        if python_executable.as_posix() != sys.executable:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_CONFLICT, "client Python executable differs from the plan"
            )

        lock_reference = _artifact_reference(_require_object(client, "dependency_lock"))
        if Path(lock_reference.path).as_posix() != (logical_run_root / "dependency-lock.json").as_posix():
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency lock path is not canonical")
        return _BootstrapInputs(
            run_id=run_id,
            logical_run_root=logical_run_root,
            shard_id=shard_id,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            scratch_root=scratch_root,
            client_runtime_sha256=runtime_sha256,
            python_abi=python_abi,
            python_executable=python_executable,
            runtime=runtime,
            dependency_lock=lock_reference,
        )

    def _verify_dependency_lock(self, inputs: _BootstrapInputs) -> _VerifiedDependencies:
        lock_bytes = read_regular_bytes(
            Path(inputs.dependency_lock.path),
            missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING,
        )
        if _sha256_bytes(lock_bytes) != inputs.dependency_lock.sha256:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_DIGEST_MISMATCH, "dependency lock digest differs")
        lock = _parse_json_bytes(lock_bytes, ClientErrorCode.DEPENDENCY_CONFLICT)
        if _require_digest(lock, "client_runtime_sha256") != inputs.client_runtime_sha256:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency lock targets another runtime")
        if _require_string(lock, "python_abi") != inputs.python_abi:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency lock targets another Python ABI")

        expected_base = _parse_distributions(lock.get("base_distributions"))
        inspected_base = _parse_distributions(inputs.runtime.get("distributions"))
        try:
            actual_base = self._inventory(None)
        except ClientWorkerError:
            raise
        except Exception as error:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_CONFLICT, "client runtime inventory cannot be verified"
            ) from error
        if expected_base != inspected_base or actual_base != expected_base:
            raise ClientWorkerError(
                ClientErrorCode.DEPENDENCY_CONFLICT, "client runtime inventory differs from the lock"
            )

        source = lock.get("source")
        if source is not None:
            source_reference = _artifact_reference(_as_object(source))
            _verify_input_artifact(
                source_reference,
                inputs.logical_run_root / "inputs",
            )
        return _VerifiedDependencies(
            base_distributions=expected_base,
            overlay_packages=tuple(_as_object_list(lock.get("overlay_packages"))),
        )

    def _prepare_overlay(
        self,
        inputs: _BootstrapInputs,
        dependencies: _VerifiedDependencies,
    ) -> tuple[Path, ClientInstallerOutcome, tuple[InstalledDistribution, ...]]:
        expected_overlay, wheels = _verify_wheels(
            dependencies.overlay_packages,
            inputs.logical_run_root / "dependencies",
            dependencies.base_distributions,
        )
        overlay_path = inputs.scratch_root / "client-env" / "site-packages"
        outcome = self._install_overlay(inputs.python_executable, wheels, expected_overlay, overlay_path)
        installed = tuple(sorted((*dependencies.base_distributions, *expected_overlay), key=lambda item: item.name))
        return overlay_path, outcome, installed

    def _install_overlay(
        self,
        python_executable: Path,
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
                    python_executable.as_posix(),
                    "-m",
                    "pip",
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
    if path is not None:
        importlib.invalidate_caches()
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
        installed.append(
            InstalledDistribution(
                name=name,
                version=version,
                provenance_sha256=(None if path is not None or direct_url is None else _direct_url_sha256(direct_url)),
            )
        )
    return tuple(sorted(installed, key=lambda item: item.name))


def activate_environment(prepared: PreparedClientEnvironment) -> None:
    """Activate one verified overlay and isolate Data Designer attempt state."""
    home = prepared.scratch_root / "data-designer-home"
    cache = prepared.scratch_root / "cache"
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


def _validate_scratch_root(scratch_root: Path, attempt_dir: Path) -> None:
    _validate_input_path(scratch_root, "allocation scratch")
    try:
        status = scratch_root.lstat()
    except OSError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "allocation scratch is unavailable") from error
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or status.st_uid != os.geteuid()
        or scratch_root == attempt_dir
        or scratch_root.is_relative_to(attempt_dir)
        or attempt_dir.is_relative_to(scratch_root)
    ):
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "allocation scratch is invalid")
    ensure_private_directory(scratch_root)


def _run_installer(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _verify_wheels(
    packages: tuple[dict[str, object], ...],
    logical_dependencies_root: Path,
    base_distributions: tuple[InstalledDistribution, ...],
) -> tuple[tuple[InstalledDistribution, ...], tuple[Path, ...]]:
    expected: list[InstalledDistribution] = []
    wheels: list[Path] = []
    base_names = {item.name for item in base_distributions}
    for package in packages:
        name = canonicalize_name(_require_string(package, "name"))
        version = _require_string(package, "version")
        if name in base_names or name in {item.name for item in expected}:
            raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "dependency distributions overlap")
        artifact = _artifact_reference(_require_object(package, "artifact"))
        logical_wheel = Path(artifact.path)
        if logical_wheel.parent != logical_dependencies_root or logical_wheel.suffix != ".whl":
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency wheel path is not canonical")
        wheel = logical_wheel
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


def _verify_input_artifact(reference: ArtifactReference, logical_root: Path) -> None:
    logical_path = Path(reference.path)
    if logical_path.parent != logical_root or logical_path.suffix != ".json":
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "dependency source path is not canonical")
    path = logical_path
    if compute_file_sha256(path, missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING) != reference.sha256:
        raise ClientWorkerError(ClientErrorCode.DEPENDENCY_DIGEST_MISMATCH, "dependency source digest differs")


def _parse_distributions(value: object) -> tuple[InstalledDistribution, ...]:
    parsed = tuple(
        InstalledDistribution(
            name=canonicalize_name(_require_string(item, "name")),
            version=_require_string(item, "version"),
            provenance_sha256=(
                None if item.get("provenance_sha256") is None else _require_digest(item, "provenance_sha256")
            ),
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
    if "dir_info" in direct_url:
        return True
    vcs_info = direct_url.get("vcs_info")
    if vcs_info is not None:
        if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
            return True
        commit_id = vcs_info.get("commit_id")
        return not isinstance(commit_id, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit_id) is None
    archive_info = direct_url.get("archive_info")
    if archive_info is None:
        return True
    if not isinstance(archive_info, dict):
        return True
    hashes = archive_info.get("hashes")
    digest = None if not isinstance(hashes, dict) else hashes.get("sha256")
    return not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None


def _direct_url_sha256(value: str) -> str:
    try:
        payload = _as_object(json.loads(value))
    except (json.JSONDecodeError, TypeError) as error:  # pragma: no cover - guarded by mutability validation
        raise ClientWorkerError(ClientErrorCode.DEPENDENCY_CONFLICT, "installed provenance is invalid") from error
    return hashlib.sha256(canonical_json(payload)).hexdigest()
