# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable records for Slurm image lifecycle and registry operations."""

from __future__ import annotations

import posixpath
from enum import Enum

from pydantic import field_validator, model_validator

from data_designer.slurm.config import ImageBuildRequest, ImageInspectionRecord, ImageKind, SelectedSlurmProfile
from data_designer.slurm.contracts import (
    ArtifactReference,
    ContractRecord,
    ContractValue,
    Identifier,
    SchemaVersion,
    Sha256Digest,
    validate_absolute_path,
)


class ImageLifecycleOperation(str, Enum):
    """Compute-node operation required by one image request."""

    IMPORT_OCI = "import_oci"
    INSPECT_SQSH = "inspect_sqsh"


class ImageLifecyclePlan(ContractRecord):
    """Immutable inputs for one CPU Slurm image lifecycle job."""

    lifecycle_id: Identifier
    request: ImageBuildRequest
    selected_profile: SelectedSlurmProfile
    operation: ImageLifecycleOperation
    job_directory: str
    sqsh_path: str
    inspection_output_path: str
    inspector_script: ArtifactReference
    enroot_rc: ArtifactReference
    source_oci_digest: Sha256Digest | None = None

    _paths_are_absolute = field_validator("job_directory", "sqsh_path", "inspection_output_path")(
        validate_absolute_path
    )

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ImageLifecyclePlan:
        validate_oci_source_for_lifecycle(self.request.source)
        workspace_root = self.selected_profile.profile.workspace_root
        expected_job_directory = posixpath.join(
            workspace_root,
            "images",
            ".tmp",
            "jobs",
            self.lifecycle_id,
        )
        if self.job_directory != expected_job_directory:
            raise ValueError("image lifecycle job directory must derive from the selected workspace")
        validate_enroot_mount_path(self.job_directory)
        if self.inspection_output_path != posixpath.join(self.job_directory, "output", "inspection.json"):
            raise ValueError("image lifecycle inspection output must belong to its dedicated output directory")
        expected_runtime_artifacts = (
            (self.inspector_script, "inspect_image.py"),
            (self.enroot_rc, "enroot.rc"),
        )
        for artifact, filename in expected_runtime_artifacts:
            if artifact.path != posixpath.join(self.job_directory, filename):
                raise ValueError("image lifecycle runtime artifacts must use their package-owned job paths")

        if self.request.source.endswith(".sqsh"):
            if self.operation is not ImageLifecycleOperation.INSPECT_SQSH:
                raise ValueError("existing SQSH requests require inspection without import")
            if self.sqsh_path != self.request.source:
                raise ValueError("existing SQSH plans must inspect the authored source path in place")
            if self.source_oci_digest is not None:
                raise ValueError("existing SQSH plans must not contain an OCI source digest")
        else:
            if self.operation is not ImageLifecycleOperation.IMPORT_OCI:
                raise ValueError("OCI requests require an import operation")
            if self.sqsh_path != posixpath.join(self.job_directory, "candidate.sqsh"):
                raise ValueError("OCI import output must remain attempt-local until publication")
            expected_source_digest = self.request.source.rpartition("@sha256:")[2]
            if self.source_oci_digest != expected_source_digest:
                raise ValueError("OCI source digest does not match the authored source")
        return self


def validate_oci_source_for_lifecycle(source: str) -> str:
    """Reject OCI source forms that could persist credentials or ambiguous schemes."""
    if source.endswith(".sqsh"):
        return source
    repository, separator, _digest = source.rpartition("@sha256:")
    if (
        separator != "@sha256:"
        or "@" in repository
        or "://" in repository
        or any(delimiter in repository for delimiter in ("?", "#"))
    ):
        raise ValueError("OCI image source must be a credential-free registry reference without a scheme")
    return source


def validate_enroot_mount_path(path: str) -> str:
    """Reject host paths that Enroot's fstab-style mount parser cannot represent safely."""
    validate_absolute_path(path)
    if any(character.isspace() or character in {":", ",", "\\"} for character in path):
        raise ValueError("image lifecycle workspace path cannot be represented as an Enroot mount")
    return path


class RegisteredImage(ContractRecord):
    """One immutable alias binding for a verified SQSH artifact."""

    name: Identifier
    path: str
    sqsh_sha256: Sha256Digest
    source_oci_digest: Sha256Digest | None = None
    inspection: ImageInspectionRecord

    @property
    def kind(self) -> ImageKind:
        """Return the factual role discovered by the package inspector."""
        return self.inspection.inspection.kind

    @property
    def immutable_facts(self) -> tuple[Sha256Digest, Sha256Digest | None, ImageInspectionRecord]:
        """Return the facts that every alias for this SQSH path must share."""
        return (self.sqsh_sha256, self.source_oci_digest, self.inspection)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        validate_absolute_path(value)
        if not value.endswith(".sqsh"):
            raise ValueError("registered image path must end in .sqsh")
        return value

    @model_validator(mode="after")
    def validate_inspection_digest(self) -> RegisteredImage:
        if self.inspection.sqsh_sha256 != self.sqsh_sha256:
            raise ValueError("image inspection digest does not match the registered SQSH")
        return self


class ImageRegistryDocument(ContractValue):
    """Versioned contents of the package-owned YAML image registry."""

    schema_version: SchemaVersion
    images: tuple[RegisteredImage, ...] = ()

    @model_validator(mode="after")
    def validate_images(self) -> ImageRegistryDocument:
        names = tuple(image.name for image in self.images)
        if names != tuple(sorted(names)):
            raise ValueError("registered images must be sorted by alias")
        if len(names) != len(set(names)):
            raise ValueError("registered image aliases must be unique")

        facts_by_path: dict[str, tuple[Sha256Digest, Sha256Digest | None, ImageInspectionRecord]] = {}
        for image in self.images:
            facts = image.immutable_facts
            existing = facts_by_path.setdefault(image.path, facts)
            if existing != facts:
                raise ValueError("aliases for one SQSH path must share identical immutable facts")
        return self
