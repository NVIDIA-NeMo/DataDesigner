# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from fnmatch import fnmatchcase
from typing import Annotated, Literal

from pydantic import Field, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.slurm.contracts import (
    AuthoredConfig,
    ContractRecord,
    Identifier,
    SchemaVersion,
    Sha256Digest,
    compute_canonical_json_sha256,
    validate_absolute_path,
    validate_plain_text,
)


class GpuRequestMode(str, Enum):
    GRES = "gres"
    VISIBLE = "visible"


class SchedulerProfile(AuthoredConfig):
    account: Identifier | None = None
    partition: Identifier | None = None
    mem_per_gpu: Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*(?:K|M|G|T)$")] | None = None


class ImageBuildProfile(AuthoredConfig):
    partition: Identifier
    cpus_per_task: PositiveInt
    memory: Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*(?:K|M|G|T)$")]
    time_limit: Annotated[str, StringConstraints(pattern=r"^[0-9]+:[0-5][0-9]:[0-5][0-9]$")]

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, value: str) -> str:
        hours, minutes, seconds = (int(component) for component in value.split(":"))
        if hours == minutes == seconds == 0:
            raise ValueError("image-build time limit must be positive")
        return value


class ContainerMount(AuthoredConfig):
    source: str
    target: str
    read_only: bool = False

    _paths_are_absolute = field_validator("source", "target")(validate_absolute_path)


class SlurmProfile(AuthoredConfig):
    """Strict facts for one Slurm cluster."""

    schema_version: SchemaVersion
    host_patterns: list[str] = Field(default_factory=list)
    scheduler: SchedulerProfile = Field(default_factory=SchedulerProfile)
    gpus_per_node: PositiveInt | Literal["auto"]
    workspace_root: str
    image_build: ImageBuildProfile
    gpu_request_mode: Literal["gres", "visible"] = "gres"
    container_mounts: list[ContainerMount] = Field(default_factory=list)

    _workspace_root_is_absolute = field_validator("workspace_root")(validate_absolute_path)

    @field_validator("host_patterns")
    @classmethod
    def validate_host_patterns(cls, values: list[str]) -> list[str]:
        normalized: set[str] = set()
        for value in values:
            validate_plain_text(value, field_name="host pattern")
            _validate_hostname_glob(value)
            pattern = value.casefold()
            if pattern in normalized:
                raise ValueError(f"duplicate hostname glob: {value!r}")
            normalized.add(pattern)
        return values

    @model_validator(mode="after")
    def validate_mounts(self) -> SlurmProfile:
        targets = [mount.target for mount in self.container_mounts]
        if len(targets) != len(set(targets)):
            raise ValueError("container mount targets must be unique")
        return self


class SlurmProfileCatalog(AuthoredConfig):
    """Versioned catalog of independently complete cluster profiles."""

    schema_version: SchemaVersion
    default_cluster: Identifier
    clusters: dict[Identifier, SlurmProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> SlurmProfileCatalog:
        if self.default_cluster not in self.clusters:
            raise ValueError("default_cluster must name a configured cluster")

        patterns: dict[str, str] = {}
        for cluster_name, profile in self.clusters.items():
            for pattern in profile.host_patterns:
                normalized = pattern.casefold()
                if normalized in patterns:
                    raise ValueError(
                        f"hostname glob {pattern!r} is duplicated by clusters "
                        f"{patterns[normalized]!r} and {cluster_name!r}"
                    )
                patterns[normalized] = cluster_name
        return self


class ProfileSelectionSource(str, Enum):
    EXPLICIT = "explicit"
    HOSTNAME = "hostname"
    DEFAULT = "default"
    INJECTED = "injected"


class SelectedSlurmProfile(ContractRecord):
    """Selected profile and provenance persisted in a resolved run plan."""

    cluster_name: Identifier | None = None
    selection_source: ProfileSelectionSource
    matched_pattern: str | None = None
    catalog_path: str | None = None
    catalog_sha256: Sha256Digest | None = None
    profile_sha256: Sha256Digest
    profile: SlurmProfile

    @field_validator("catalog_path")
    @classmethod
    def validate_catalog_path(cls, value: str | None) -> str | None:
        return None if value is None else validate_absolute_path(value)

    @model_validator(mode="after")
    def validate_selection(self) -> SelectedSlurmProfile:
        if self.profile_sha256 != _profile_digest(self.profile):
            raise ValueError("profile_sha256 does not match the selected profile")

        catalog_fields = (self.cluster_name, self.catalog_sha256)
        if self.selection_source is ProfileSelectionSource.INJECTED:
            if any(value is not None for value in (*catalog_fields, self.catalog_path, self.matched_pattern)):
                raise ValueError("injected profiles must not contain catalog selection fields")
        else:
            if any(value is None for value in catalog_fields):
                raise ValueError("catalog selections require cluster_name and catalog_sha256")
            if self.selection_source is ProfileSelectionSource.HOSTNAME:
                if self.matched_pattern is None:
                    raise ValueError("hostname selection requires matched_pattern")
            elif self.matched_pattern is not None:
                raise ValueError("only hostname selection may contain matched_pattern")
        return self


def select_profile(
    catalog: SlurmProfileCatalog,
    *,
    cluster: str | None = None,
    hostnames: tuple[str, ...] = (),
    catalog_path: str | None = None,
) -> SelectedSlurmProfile:
    """Select a profile with explicit, hostname, then default precedence."""
    catalog_sha256 = compute_canonical_json_sha256(catalog.model_dump(mode="json"))
    if cluster is not None:
        if cluster not in catalog.clusters:
            raise ValueError(f"unknown cluster {cluster!r}")
        return _selection(
            catalog,
            cluster,
            ProfileSelectionSource.EXPLICIT,
            catalog_sha256,
            catalog_path=catalog_path,
        )

    normalized_hosts = {hostname.casefold() for hostname in hostnames if hostname}
    matches: dict[str, list[str]] = {}
    for cluster_name, profile in catalog.clusters.items():
        matching_patterns = sorted(
            pattern
            for pattern in profile.host_patterns
            if any(fnmatchcase(hostname, pattern.casefold()) for hostname in normalized_hosts)
        )
        if matching_patterns:
            matches[cluster_name] = matching_patterns

    if len(matches) > 1:
        raise ValueError(f"hostname matches multiple clusters: {', '.join(sorted(matches))}")
    if matches:
        selected_name = next(iter(matches))
        return _selection(
            catalog,
            selected_name,
            ProfileSelectionSource.HOSTNAME,
            catalog_sha256,
            catalog_path=catalog_path,
            matched_pattern=matches[selected_name][0],
        )
    return _selection(
        catalog,
        catalog.default_cluster,
        ProfileSelectionSource.DEFAULT,
        catalog_sha256,
        catalog_path=catalog_path,
    )


def injected_profile(profile: SlurmProfile) -> SelectedSlurmProfile:
    """Record a directly injected effective profile."""
    return SelectedSlurmProfile(
        schema_version=1,
        selection_source=ProfileSelectionSource.INJECTED,
        profile_sha256=_profile_digest(profile),
        profile=profile,
    )


def validate_selected_profile(
    catalog: SlurmProfileCatalog,
    selected: SelectedSlurmProfile,
) -> SelectedSlurmProfile:
    """Validate a persisted catalog selection against its source catalog."""
    if selected.selection_source is ProfileSelectionSource.INJECTED:
        raise ValueError("injected profile selection has no source catalog")
    if selected.catalog_sha256 != compute_canonical_json_sha256(catalog.model_dump(mode="json")):
        raise ValueError("selected profile catalog digest does not match the catalog")
    if selected.cluster_name not in catalog.clusters:
        raise ValueError("selected cluster is absent from the catalog")
    if selected.profile != catalog.clusters[selected.cluster_name]:
        raise ValueError("selected profile does not match its catalog entry")
    if selected.selection_source is ProfileSelectionSource.DEFAULT and selected.cluster_name != catalog.default_cluster:
        raise ValueError("default profile selection does not match the catalog default")
    if (
        selected.selection_source is ProfileSelectionSource.HOSTNAME
        and selected.matched_pattern not in selected.profile.host_patterns
    ):
        raise ValueError("hostname selection pattern is absent from the selected profile")
    return selected


def _selection(
    catalog: SlurmProfileCatalog,
    cluster_name: str,
    source: ProfileSelectionSource,
    catalog_sha256: Sha256Digest,
    *,
    catalog_path: str | None,
    matched_pattern: str | None = None,
) -> SelectedSlurmProfile:
    profile = catalog.clusters[cluster_name]
    return SelectedSlurmProfile(
        schema_version=1,
        cluster_name=cluster_name,
        selection_source=source,
        matched_pattern=matched_pattern,
        catalog_path=catalog_path,
        catalog_sha256=catalog_sha256,
        profile_sha256=_profile_digest(profile),
        profile=profile,
    )


def _profile_digest(profile: SlurmProfile) -> Sha256Digest:
    return compute_canonical_json_sha256(profile.model_dump(mode="json"))


def _validate_hostname_glob(value: str) -> None:
    if "/" in value or any(character.isspace() for character in value):
        raise ValueError(f"invalid hostname glob: {value!r}")
    open_class: int | None = None
    for index, character in enumerate(value):
        if character == "[":
            if open_class is not None:
                raise ValueError(f"invalid hostname glob: {value!r}")
            open_class = index
        elif character == "]":
            if open_class is None or index == open_class + 1:
                raise ValueError(f"invalid hostname glob: {value!r}")
            open_class = None
    if open_class is not None:
        raise ValueError(f"invalid hostname glob: {value!r}")
