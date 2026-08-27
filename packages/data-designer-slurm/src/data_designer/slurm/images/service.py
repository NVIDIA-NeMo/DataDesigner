# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified existing-image registry operations for Slurm planning."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from data_designer.slurm.config import ImageBuildRequest, ImageInspectionRecord, ImageKind, ImageRef
from data_designer.slurm.contracts import Sha256Digest
from data_designer.slurm.images.errors import ImageVerificationError
from data_designer.slurm.images.inspection import INSPECTOR_VERSION
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.images.registry import ImageRegistryStore
from data_designer.slurm.planning import ResolvedImage

_HASH_BLOCK_SIZE = 1024 * 1024


class VerifiedImageRegistry:
    """Provide verified existing-SQSH operations for the public image facade."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._registry = ImageRegistryStore(workspace_root)

    def register_existing(
        self,
        request: ImageBuildRequest,
        inspection: ImageInspectionRecord,
        *,
        replace: bool = False,
    ) -> RegisteredImage:
        """Verify and register an existing compute-visible SQSH in place."""
        if not request.source.endswith(".sqsh"):
            raise ImageVerificationError("OCI image sources require the CPU Slurm image lifecycle")
        path = Path(request.source)
        _validate_inspection(inspection, expected_kind=ImageKind(request.kind))
        image = RegisteredImage(
            schema_version=1,
            name=request.name,
            path=path.as_posix(),
            sqsh_sha256=inspection.sqsh_sha256,
            inspection=inspection,
        )
        return self._registry.register(image, verify_before_publish=_verify_registered_image, replace=replace)

    def list_images(self) -> tuple[RegisteredImage, ...]:
        """Return all registered aliases in deterministic order."""
        return self._registry.list_images()

    def resolve_for_planning(self, reference: ImageRef, *, expected_kind: ImageKind) -> ResolvedImage:
        """Resolve and reverify one registered alias or path for plan compilation."""
        if reference.name is not None:
            image = self._registry.get_by_name(reference.name)
        else:
            assert reference.path is not None
            image = self._registry.get_by_path(reference.path)
        _verify_registered_image(image)
        _validate_inspection(
            image.inspection,
            expected_kind=expected_kind,
            expected_sha256=image.sqsh_sha256,
        )
        return ResolvedImage(
            authored_ref=reference,
            path=image.path,
            sha256=image.sqsh_sha256,
            inspection=image.inspection,
        )

    def unregister(self, name: str) -> RegisteredImage:
        """Remove an alias while leaving its underlying artifact untouched."""
        return self._registry.unregister(name)


def compute_sqsh_file_sha256(path: Path) -> Sha256Digest:
    """Compute the SHA-256 of one regular, non-symlink SQSH file."""
    descriptor: int | None = None
    try:
        before_open = path.lstat()
        if not stat.S_ISREG(before_open.st_mode):
            raise ImageVerificationError(f"image path {path} must be a regular non-symlink file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        image_file = os.fdopen(descriptor, "rb")
        descriptor = None
        digest = hashlib.sha256()
        with image_file:
            before_read = os.fstat(image_file.fileno())
            if (before_open.st_dev, before_open.st_ino) != (before_read.st_dev, before_read.st_ino):
                raise ImageVerificationError(f"image path {path} changed while it was being opened")
            while block := image_file.read(_HASH_BLOCK_SIZE):
                digest.update(block)
            after_read = os.fstat(image_file.fileno())
            after_path = path.lstat()
        before_facts = (
            before_read.st_dev,
            before_read.st_ino,
            before_read.st_size,
            before_read.st_mtime_ns,
            before_read.st_ctime_ns,
        )
        after_facts = (
            after_read.st_dev,
            after_read.st_ino,
            after_read.st_size,
            after_read.st_mtime_ns,
            after_read.st_ctime_ns,
        )
        if before_facts != after_facts:
            raise ImageVerificationError(f"image path {path} changed while it was being verified")
        path_facts = (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        )
        if not stat.S_ISREG(after_path.st_mode) or path_facts != after_facts:
            raise ImageVerificationError(f"image path {path} changed while it was being verified")
        return digest.hexdigest()
    except FileNotFoundError as error:
        raise ImageVerificationError(f"image path {path} does not exist") from error
    except OSError as error:
        raise ImageVerificationError(f"cannot read image path {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_inspection(
    inspection: ImageInspectionRecord,
    *,
    expected_kind: ImageKind,
    expected_sha256: Sha256Digest | None = None,
) -> None:
    if inspection.inspector_version != INSPECTOR_VERSION:
        raise ImageVerificationError(f"unsupported image inspector version {inspection.inspector_version!r}")
    if expected_sha256 is not None and inspection.sqsh_sha256 != expected_sha256:
        raise ImageVerificationError("image inspection does not match the SQSH digest")
    if inspection.inspection.kind is not expected_kind:
        raise ImageVerificationError(
            f"image inspection kind {inspection.inspection.kind.value!r} does not match {expected_kind.value!r}"
        )


def _verify_registered_image(image: RegisteredImage) -> None:
    actual_sha256 = compute_sqsh_file_sha256(Path(image.path))
    if actual_sha256 != image.sqsh_sha256:
        raise ImageVerificationError(f"registered image {image.name!r} no longer matches its SQSH digest")
