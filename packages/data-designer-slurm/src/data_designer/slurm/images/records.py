# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable records persisted by the Slurm image registry."""

from __future__ import annotations

from pydantic import field_validator, model_validator

from data_designer.slurm.config import ImageInspectionRecord, ImageKind
from data_designer.slurm.contracts import (
    ContractRecord,
    ContractValue,
    Identifier,
    SchemaVersion,
    Sha256Digest,
    validate_absolute_path,
)


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
