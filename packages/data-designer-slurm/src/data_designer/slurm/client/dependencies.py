# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve immutable client dependency locks before Slurm submission."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename

from data_designer.slurm.client.filesystem import compute_file_sha256, read_regular_bytes
from data_designer.slurm.client.records import ClientErrorCode
from data_designer.slurm.config import ClientDependencies, ClientImageInspection
from data_designer.slurm.contracts import ArtifactReference, InstalledDistribution, validate_absolute_path
from data_designer.slurm.planning import LockedPackage, ResolvedDependencyLock, ResolvedImage

_RESOLVER_VERSION = "pip-pure-wheel-1"
_DEFAULT_INDEX_URL = "https://pypi.org/simple"
_LOCK_SOURCE_DIRECTORY = "inputs"
_DEPENDENCY_DIRECTORY = "dependencies"

DependencyCommandRunner = Callable[[tuple[str, ...], Path, Mapping[str, str]], None]


class ClientDependencyResolutionError(RuntimeError):
    """Raised when authored client dependencies cannot produce a safe lock."""


@dataclass(frozen=True, slots=True)
class ResolvedClientDependencies:
    """Resolved lock plus source artifacts for atomic run initialization."""

    lock: ResolvedDependencyLock
    wheel_sources: tuple[Path, ...]
    lock_source: Path | None = None


class ClientDependencyResolver:
    """Resolve pure wheels against the immutable client-image inventory."""

    def __init__(
        self,
        *,
        command_runner: DependencyCommandRunner | None = None,
        environ: Mapping[str, str] | None = None,
        python_executable: str | None = None,
    ) -> None:
        self._command_runner = command_runner or _run_pip_download
        self._environ = os.environ if environ is None else environ
        self._python_executable = sys.executable if python_executable is None else python_executable

    @contextmanager
    def resolve(
        self,
        dependencies: ClientDependencies,
        client_image: ResolvedImage,
        *,
        run_root: str | Path,
        source_root: str | Path,
    ) -> Iterator[ResolvedClientDependencies]:
        """Yield one verified resolution while temporary wheel sources remain live."""
        if not isinstance(dependencies, ClientDependencies):
            raise ClientDependencyResolutionError("client dependencies are invalid")
        inspection = client_image.inspection_facts
        if not isinstance(inspection, ClientImageInspection):
            raise ClientDependencyResolutionError("client image lacks dependency inspection facts")
        try:
            normalized_run_root = Path(validate_absolute_path(Path(run_root).as_posix()))
            normalized_source_root = Path(validate_absolute_path(Path(source_root).as_posix()))
        except ValueError as error:
            raise ClientDependencyResolutionError("dependency resolution paths are invalid") from error

        with tempfile.TemporaryDirectory(prefix="data-designer-slurm-dependencies-") as temporary:
            temporary_root = Path(temporary).resolve()
            try:
                if dependencies.requirements is None:
                    result = self._load_lock(
                        dependencies,
                        client_image,
                        normalized_run_root,
                        normalized_source_root,
                    )
                else:
                    result = self._resolve_requirements(
                        dependencies,
                        client_image,
                        normalized_run_root,
                        temporary_root,
                    )
            except ClientDependencyResolutionError:
                raise
            except OSError as error:
                raise ClientDependencyResolutionError("client dependency artifacts are unavailable") from error
            yield result

    def _resolve_requirements(
        self,
        dependencies: ClientDependencies,
        client_image: ResolvedImage,
        run_root: Path,
        temporary_root: Path,
    ) -> ResolvedClientDependencies:
        requirements = dependencies.requirements
        if requirements is None:
            raise ClientDependencyResolutionError("client dependency requirements are unavailable")
        inspection = client_image.inspection_facts
        if not isinstance(inspection, ClientImageInspection):
            raise ClientDependencyResolutionError("client image lacks dependency inspection facts")
        image_distributions = tuple(sorted(inspection.distributions, key=lambda item: item.name))
        if not requirements:
            return ResolvedClientDependencies(
                lock=_build_lock(client_image, image_distributions=image_distributions, authored_requirements=()),
                wheel_sources=(),
            )

        requirements_path = temporary_root / "requirements.in"
        constraints_path = temporary_root / "constraints.txt"
        wheelhouse = temporary_root / "wheels"
        wheelhouse.mkdir(mode=0o700)
        requirements_path.write_text("".join(f"{item}\n" for item in requirements), encoding="utf-8")
        constraints_path.write_text(
            "".join(f"{item.name}=={item.version}\n" for item in image_distributions),
            encoding="utf-8",
        )
        command = (
            self._python_executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--no-input",
            f"--index-url={_DEFAULT_INDEX_URL}",
            "--only-binary=:all:",
            "--platform=any",
            "--implementation=py",
            "--abi=none",
            f"--python-version={inspection.python_version.rsplit('.', maxsplit=1)[0]}",
            f"--dest={wheelhouse.as_posix()}",
            f"--requirement={requirements_path.as_posix()}",
            f"--constraint={constraints_path.as_posix()}",
        )
        environment = self._resolution_environment(dependencies)
        try:
            self._command_runner(command, temporary_root, environment)
        except ClientDependencyResolutionError:
            raise
        except Exception as error:
            raise ClientDependencyResolutionError("client dependency resolution failed") from error

        packages, sources = _load_downloaded_wheels(wheelhouse, run_root, image_distributions)
        try:
            lock = _build_lock(
                client_image,
                image_distributions=image_distributions,
                authored_requirements=tuple(requirements),
                overlay_packages=packages,
            )
        except ValueError as error:
            raise ClientDependencyResolutionError("resolved client dependencies are incompatible") from error
        return ResolvedClientDependencies(lock=lock, wheel_sources=sources)

    def _load_lock(
        self,
        dependencies: ClientDependencies,
        client_image: ResolvedImage,
        run_root: Path,
        source_root: Path,
    ) -> ResolvedClientDependencies:
        lock_file = dependencies.lock_file
        if lock_file is None:
            raise ClientDependencyResolutionError("client dependency lock is unavailable")
        source = source_root / lock_file
        try:
            content = read_regular_bytes(source, missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING)
            supplied = ResolvedDependencyLock.model_validate_json(content, strict=True)
        except Exception as error:
            raise ClientDependencyResolutionError("client dependency lock is invalid") from error
        inspection = client_image.inspection_facts
        if not isinstance(inspection, ClientImageInspection):
            raise ClientDependencyResolutionError("client image lacks dependency inspection facts")
        if (
            supplied.client_image_sha256 != client_image.sha256
            or supplied.python_abi != inspection.python_abi
            or supplied.image_distributions != tuple(sorted(inspection.distributions, key=lambda item: item.name))
        ):
            raise ClientDependencyResolutionError("client dependency lock targets another image")

        packages: list[LockedPackage] = []
        sources: list[Path] = []
        for package in supplied.overlay_packages:
            wheel = Path(package.artifact.path)
            _verify_locked_wheel(wheel, package)
            packages.append(
                LockedPackage(
                    name=package.name,
                    version=package.version,
                    artifact=ArtifactReference(
                        path=(run_root / _DEPENDENCY_DIRECTORY / wheel.name).as_posix(),
                        sha256=package.artifact.sha256,
                    ),
                )
            )
            sources.append(wheel)
        source_reference = ArtifactReference(
            path=(run_root / _LOCK_SOURCE_DIRECTORY / source.name).as_posix(),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        try:
            lock = ResolvedDependencyLock(
                schema_version=1,
                resolver_version=supplied.resolver_version,
                python_abi=supplied.python_abi,
                client_image_sha256=supplied.client_image_sha256,
                authored_requirements=supplied.authored_requirements,
                authored_source=lock_file,
                source=source_reference,
                image_distributions=supplied.image_distributions,
                overlay_packages=tuple(packages),
            )
        except ValueError as error:
            raise ClientDependencyResolutionError("client dependency lock is incompatible") from error
        return ResolvedClientDependencies(lock=lock, wheel_sources=tuple(sources), lock_source=source)

    def _resolution_environment(self, dependencies: ClientDependencies) -> dict[str, str]:
        credential_names = {reference.environment for reference in dependencies.index_credentials.values()}
        for reference in dependencies.index_credentials.values():
            value = self._environ.get(reference.environment)
            if value is None or not value:
                raise ClientDependencyResolutionError("client dependency index credential is unavailable")
        # These credentials belong to the allocation runtime, not submit-side pip.
        environment = {
            name: value
            for name, value in self._environ.items()
            if not name.startswith("PIP_") and name not in credential_names
        }
        environment["PIP_CONFIG_FILE"] = os.devnull
        return environment


def _build_lock(
    client_image: ResolvedImage,
    *,
    image_distributions: tuple[InstalledDistribution, ...],
    authored_requirements: tuple[str, ...],
    overlay_packages: tuple[LockedPackage, ...] = (),
) -> ResolvedDependencyLock:
    inspection = client_image.inspection_facts
    if not isinstance(inspection, ClientImageInspection):
        raise ClientDependencyResolutionError("client image lacks dependency inspection facts")
    return ResolvedDependencyLock(
        schema_version=1,
        resolver_version=_RESOLVER_VERSION,
        python_abi=inspection.python_abi,
        client_image_sha256=client_image.sha256,
        authored_requirements=authored_requirements,
        image_distributions=image_distributions,
        overlay_packages=overlay_packages,
    )


def _load_downloaded_wheels(
    wheelhouse: Path,
    run_root: Path,
    image_distributions: tuple[InstalledDistribution, ...],
) -> tuple[tuple[LockedPackage, ...], tuple[Path, ...]]:
    image = {item.name: item.version for item in image_distributions}
    packages: list[tuple[LockedPackage, Path]] = []
    names: set[str] = set()
    for wheel in sorted(wheelhouse.iterdir(), key=lambda path: path.name):
        status = wheel.lstat()
        if not stat.S_ISREG(status.st_mode) or wheel.suffix != ".whl":
            raise ClientDependencyResolutionError("dependency resolver produced an invalid artifact")
        try:
            name, version, _, tags = parse_wheel_filename(wheel.name)
        except ValueError as error:
            raise ClientDependencyResolutionError("dependency resolver produced an invalid wheel") from error
        normalized_name = canonicalize_name(name)
        normalized_version = str(version)
        if not tags or any(tag.abi != "none" or tag.platform != "any" for tag in tags):
            raise ClientDependencyResolutionError("client dependencies must resolve to platform-independent wheels")
        if normalized_name in image:
            if image[normalized_name] != normalized_version:
                raise ClientDependencyResolutionError("client dependency conflicts with the selected image")
            continue
        if normalized_name in names:
            raise ClientDependencyResolutionError("dependency resolver produced duplicate packages")
        names.add(normalized_name)
        digest = compute_file_sha256(wheel, missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING)
        packages.append(
            (
                LockedPackage(
                    name=normalized_name,
                    version=normalized_version,
                    artifact=ArtifactReference(
                        path=(run_root / _DEPENDENCY_DIRECTORY / wheel.name).as_posix(),
                        sha256=digest,
                    ),
                ),
                wheel,
            )
        )
    packages.sort(key=lambda item: item[0].name)
    return tuple(item[0] for item in packages), tuple(item[1] for item in packages)


def _verify_locked_wheel(wheel: Path, package: LockedPackage) -> None:
    try:
        name, version, _, tags = parse_wheel_filename(wheel.name)
        digest = compute_file_sha256(wheel, missing_code=ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING)
    except Exception as error:
        raise ClientDependencyResolutionError("client dependency artifact is invalid") from error
    if (
        not tags
        or any(tag.abi != "none" or tag.platform != "any" for tag in tags)
        or canonicalize_name(name) != package.name
        or str(version) != package.version
        or digest != package.artifact.sha256
    ):
        raise ClientDependencyResolutionError("client dependency artifact differs from its lock")


def _run_pip_download(command: tuple[str, ...], cwd: Path, environment: Mapping[str, str]) -> None:
    try:
        subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ClientDependencyResolutionError("client dependency resolution failed") from error


__all__ = [
    "ClientDependencyResolutionError",
    "ClientDependencyResolver",
    "ResolvedClientDependencies",
]
