# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-owned Slurm profile initialization and validation."""

from __future__ import annotations

import json
import os
import shlex
import socket
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path

import yaml
from pydantic import PositiveInt, ValidationError

from data_designer.slurm.config import (
    DEFAULT_PROFILE_FILE_NAME,
    PROFILE_FILE_ENVIRONMENT,
    ProfileSelectionSource,
    SlurmConfigLoadError,
    SlurmProfile,
    SlurmProfileCatalog,
    load_profile_catalog,
    resolve_profile,
)
from data_designer.slurm.contracts import ContractValue, Identifier, compute_canonical_json_sha256
from data_designer.slurm.filesystem import create_restrictive_temporary_file, open_verified_directory
from data_designer.slurm.images.registry import ImageRegistryStore
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.errors import SlurmLauncherError
from data_designer.slurm.services.errors import SlurmServiceError, SlurmServiceErrorCode, SlurmServiceOperation

HostnameResolver = Callable[[], tuple[str, ...]]

_IMAGE_BUILD_CPUS = 2
_IMAGE_BUILD_MEMORY = "8G"
_IMAGE_BUILD_TIME_LIMIT = "04:00:00"


class SlurmProfileInitialization(ContractValue):
    """One newly created profile catalog."""

    profile_file: str
    validation_command: str


class SlurmProfileMatch(ContractValue):
    """One cluster and its matching hostname patterns."""

    cluster: Identifier
    patterns: tuple[str, ...]


class SlurmProfileValidation(ContractValue):
    """Effective local profile selection and workspace facts."""

    profile_file: str | None
    hostnames: tuple[str, ...]
    matched_clusters: tuple[SlurmProfileMatch, ...]
    default_cluster: Identifier | None
    selected_cluster: Identifier | None
    selection_source: ProfileSelectionSource
    matched_pattern: str | None
    workspace_root: str
    image_root: str
    registry_file: str
    gpus_per_node: PositiveInt


class SlurmProfileService:
    """Initialize and validate profiles through production loaders and selectors."""

    def __init__(
        self,
        *,
        profile: SlurmProfile | None = None,
        catalog: SlurmProfileCatalog | None = None,
        profile_file: str | Path | None = None,
        cluster: str | None = None,
        launcher: SlurmCommandClient | None = None,
        hostname_resolver: HostnameResolver | None = None,
        environ: Mapping[str, str] | None = None,
        home_directory: str | Path | None = None,
    ) -> None:
        self._profile = profile
        self._catalog = catalog
        self._profile_file = profile_file
        self._cluster = cluster
        self._launcher = launcher or SlurmCommandClient()
        self._hostname_resolver = hostname_resolver or _local_hostnames
        self._environ = dict(os.environ if environ is None else environ)
        self._home_directory = home_directory

    def initialize(
        self,
        *,
        workspace_root: str | Path,
        image_build_partition: str,
        cluster: str = "default",
        account: str | None = None,
        partition: str | None = None,
        host_patterns: tuple[str, ...] = (),
    ) -> SlurmProfileInitialization:
        """Create one deterministic starter catalog without overwriting a file."""
        operation = SlurmServiceOperation.INIT_PROFILE
        try:
            path = _resolve_profile_path(
                self._profile_file,
                environ=self._environ,
                home_directory=self._home_directory,
            )
            selected_patterns = host_patterns or _normalize_hostnames(self._hostname_resolver())
            payload = _starter_catalog_payload(
                workspace_root=Path(workspace_root).expanduser().resolve(),
                image_build_partition=image_build_partition,
                cluster=cluster,
                account=account,
                partition=partition,
                host_patterns=selected_patterns,
            )
            SlurmProfileCatalog.model_validate(payload, strict=True)
            _create_profile_file(path, _serialize_catalog(payload, suffix=path.suffix))
        except FileExistsError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                operation,
                "profile file already exists",
            ) from None
        except FileNotFoundError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "profile destination parent does not exist",
            ) from None
        except (SlurmConfigLoadError, ValidationError, ValueError, TypeError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "profile initialization input is invalid",
            ) from None
        except OSError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                operation,
                "profile file cannot be created",
            ) from None
        command = shlex.join(("data-designer", "slurm", "profile", "validate", "--profile-file", path.as_posix()))
        return SlurmProfileInitialization(profile_file=path.as_posix(), validation_command=command)

    def validate(self) -> SlurmProfileValidation:
        """Load, select, and verify one effective production profile."""
        operation = SlurmServiceOperation.VALIDATE_PROFILE
        try:
            hostnames = _normalize_hostnames(self._hostname_resolver())
            selected = resolve_profile(
                profile=self._profile,
                catalog=self._catalog,
                profile_file=self._profile_file,
                cluster=self._cluster,
                hostnames=hostnames,
                environ=self._environ,
                home_directory=self._home_directory,
            )
            catalog = self._catalog
            if selected.catalog_path is not None:
                catalog = load_profile_catalog(selected.catalog_path)
                if compute_canonical_json_sha256(catalog.model_dump(mode="json")) != selected.catalog_sha256:
                    raise SlurmConfigLoadError("profile catalog changed while it was being validated")
            _validate_workspace(Path(selected.profile.workspace_root))
            gpus_per_node = self._resolve_gpu_count(selected.profile)
        except SlurmServiceError:
            raise
        except SlurmLauncherError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                operation,
                "Slurm is unavailable",
            ) from None
        except (SlurmConfigLoadError, ValidationError, ValueError, TypeError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "profile configuration cannot be resolved",
            ) from None
        except OSError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                operation,
                "profile workspace is unavailable",
            ) from None

        store = ImageRegistryStore(selected.profile.workspace_root)
        return SlurmProfileValidation(
            profile_file=selected.catalog_path,
            hostnames=hostnames,
            matched_clusters=_matching_clusters(catalog, hostnames),
            default_cluster=None if catalog is None else catalog.default_cluster,
            selected_cluster=selected.cluster_name,
            selection_source=selected.selection_source,
            matched_pattern=selected.matched_pattern,
            workspace_root=selected.profile.workspace_root,
            image_root=store.image_root.as_posix(),
            registry_file=store.registry_path.as_posix(),
            gpus_per_node=gpus_per_node,
        )

    def _resolve_gpu_count(self, profile: SlurmProfile) -> int:
        if profile.gpus_per_node != "auto":
            return profile.gpus_per_node
        counts = tuple(sorted(set(self._launcher.query_gpu_counts(partition=profile.scheduler.partition))))
        if len(counts) != 1:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                SlurmServiceOperation.VALIDATE_PROFILE,
                "eligible Slurm nodes do not report one GPU count",
            )
        return counts[0]


def create_slurm_profile_service(
    *,
    profile: SlurmProfile | None = None,
    catalog: SlurmProfileCatalog | None = None,
    profile_file: str | Path | None = None,
    cluster: str | None = None,
    launcher: SlurmCommandClient | None = None,
    hostname_resolver: HostnameResolver | None = None,
    environ: Mapping[str, str] | None = None,
    home_directory: str | Path | None = None,
) -> SlurmProfileService:
    """Create the package-owned profile service."""
    return SlurmProfileService(
        profile=profile,
        catalog=catalog,
        profile_file=profile_file,
        cluster=cluster,
        launcher=launcher,
        hostname_resolver=hostname_resolver,
        environ=environ,
        home_directory=home_directory,
    )


def _starter_catalog_payload(
    *,
    workspace_root: Path,
    image_build_partition: str,
    cluster: str,
    account: str | None,
    partition: str | None,
    host_patterns: tuple[str, ...],
) -> dict[str, object]:
    scheduler = {key: value for key, value in (("account", account), ("partition", partition)) if value is not None}
    profile: dict[str, object] = {
        "schema_version": 1,
        "host_patterns": list(host_patterns),
        "gpus_per_node": "auto",
        "workspace_root": workspace_root.as_posix(),
        "image_build": {
            "partition": image_build_partition,
            "cpus_per_task": _IMAGE_BUILD_CPUS,
            "memory": _IMAGE_BUILD_MEMORY,
            "time_limit": _IMAGE_BUILD_TIME_LIMIT,
        },
        "gpu_request_mode": "gres",
    }
    if scheduler:
        profile["scheduler"] = scheduler
    return {
        "schema_version": 1,
        "default_cluster": cluster,
        "clusters": {cluster: profile},
    }


def _serialize_catalog(payload: dict[str, object], *, suffix: str) -> bytes:
    if suffix == ".json":
        return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode()


def _create_profile_file(path: Path, content: bytes) -> None:
    with open_verified_directory(path.parent, resource_name="profile") as parent_descriptor:
        descriptor, temporary_name = create_restrictive_temporary_file(
            parent_descriptor,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.fsync(parent_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass


def _validate_workspace(path: Path) -> None:
    with open_verified_directory(path, resource_name="profile workspace") as descriptor:
        temporary_descriptor, temporary_name = create_restrictive_temporary_file(
            descriptor,
            prefix=".data-designer-profile-validation-",
            suffix=".tmp",
        )
        try:
            os.close(temporary_descriptor)
            temporary_descriptor = -1
            os.unlink(temporary_name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            if temporary_descriptor >= 0:
                os.close(temporary_descriptor)
            try:
                os.unlink(temporary_name, dir_fd=descriptor)
            except FileNotFoundError:
                pass


def _matching_clusters(
    catalog: SlurmProfileCatalog | None,
    hostnames: tuple[str, ...],
) -> tuple[SlurmProfileMatch, ...]:
    if catalog is None:
        return ()
    matches = []
    for cluster, profile in sorted(catalog.clusters.items()):
        patterns = tuple(
            sorted(
                pattern
                for pattern in profile.host_patterns
                if any(fnmatchcase(hostname, pattern.casefold()) for hostname in hostnames)
            )
        )
        if patterns:
            matches.append(SlurmProfileMatch(cluster=cluster, patterns=patterns))
    return tuple(matches)


def _resolve_profile_path(
    explicit_path: str | Path | None,
    *,
    environ: Mapping[str, str],
    home_directory: str | Path | None,
) -> Path:
    source = explicit_path
    if source is None:
        source = environ.get(PROFILE_FILE_ENVIRONMENT)
        if source == "":
            raise SlurmConfigLoadError(f"{PROFILE_FILE_ENVIRONMENT} must not be empty")
    if source is None:
        home = Path.home() if home_directory is None else Path(home_directory)
        source = home / DEFAULT_PROFILE_FILE_NAME
    expanded = Path(source).expanduser()
    path = expanded.parent.resolve() / expanded.name
    if path.suffix not in {".json", ".yaml", ".yml"}:
        raise SlurmConfigLoadError("configuration path must end in .json, .yaml, or .yml")
    return path


def _normalize_hostnames(hostnames: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(hostname.strip().casefold() for hostname in hostnames if hostname.strip()))


def _local_hostnames() -> tuple[str, ...]:
    return socket.gethostname(), socket.getfqdn()


__all__ = [
    "HostnameResolver",
    "SlurmProfileInitialization",
    "SlurmProfileMatch",
    "SlurmProfileService",
    "SlurmProfileValidation",
    "create_slurm_profile_service",
]
