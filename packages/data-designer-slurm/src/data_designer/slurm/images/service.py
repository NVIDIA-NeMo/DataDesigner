# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified existing-image registry operations for Slurm planning."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from data_designer.slurm.config import ImageBuildRequest, ImageInspectionRecord, ImageKind, ImageRef
from data_designer.slurm.contracts import Sha256Digest
from data_designer.slurm.images.errors import ImageConflictError, ImageVerificationError
from data_designer.slurm.images.filesystem import ensure_private_directory, open_verified_directory
from data_designer.slurm.images.inspection import INSPECTOR_VERSION
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.images.registry import ImageRegistryStore
from data_designer.slurm.planning import ResolvedImage

_HASH_BLOCK_SIZE = 1024 * 1024
_ARTIFACT_DIRECTORY_NAME = "artifacts"
_TEMPORARY_DIRECTORY_NAME = ".tmp"


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
        with _open_verified_sqsh(path) as snapshot:
            _validate_inspection(
                inspection,
                expected_kind=ImageKind(request.kind),
                expected_sha256=snapshot.sha256,
            )
            image = RegisteredImage(
                schema_version=1,
                name=request.name,
                path=path.as_posix(),
                sqsh_sha256=inspection.sqsh_sha256,
                inspection=inspection,
            )
            verify = lambda _image: _verify_snapshot(image, snapshot)
            return self._registry.register(
                image,
                verify_before_publish=verify,
                verify_after_publish=verify,
                replace=replace,
            )

    def publish_imported(
        self,
        request: ImageBuildRequest,
        inspection: ImageInspectionRecord,
        candidate_path: Path,
        *,
        source_oci_digest: Sha256Digest,
        replace: bool = False,
    ) -> RegisteredImage:
        """Publish one verified lifecycle-owned OCI import and register its alias.

        The candidate must live in this workspace's lifecycle temporary tree.
        Publication keeps a prior alias or artifact intact unless replacement is
        explicit and the new artifact is fully verified.

        Args:
            request: Authored digest-pinned OCI image request.
            inspection: Package-owned facts bound to the imported SQSH digest.
            candidate_path: Lifecycle-owned SQSH candidate beneath workspace temporary storage.
            source_oci_digest: Resolved source digest carried by the lifecycle plan.
            replace: Whether an existing alias may be replaced explicitly.

        Returns:
            The durable alias binding for the published artifact.

        Raises:
            ImageConflictError: If an alias or artifact collision is not identical.
            ImageVerificationError: If source, candidate, inspection, or publication verification fails.
            ImageRegistryError: If registry publication or recovery fails.
        """
        if request.source.endswith(".sqsh"):
            raise ImageVerificationError("existing SQSH sources must be registered in place")
        expected_source_digest = request.source.rpartition("@sha256:")[2]
        if source_oci_digest != expected_source_digest:
            raise ImageVerificationError("OCI source digest does not match the authored source")
        temporary_root = self._registry.image_root / _TEMPORARY_DIRECTORY_NAME
        if not candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise ImageVerificationError("imported SQSH candidate must belong to lifecycle temporary storage")
        try:
            candidate_path.relative_to(temporary_root)
        except ValueError as error:
            raise ImageVerificationError(
                "imported SQSH candidate must belong to lifecycle temporary storage"
            ) from error
        _validate_inspection(inspection, expected_kind=ImageKind(request.kind))
        artifact_directory = self._registry.image_root / _ARTIFACT_DIRECTORY_NAME
        artifact_path = artifact_directory / f"{request.name}-{inspection.sqsh_sha256}.sqsh"
        image = RegisteredImage(
            schema_version=1,
            name=request.name,
            path=artifact_path.as_posix(),
            sqsh_sha256=inspection.sqsh_sha256,
            source_oci_digest=source_oci_digest,
            inspection=inspection,
        )
        created_artifact_identity: tuple[int, int] | None = None
        with _open_verified_sqsh(candidate_path) as candidate_snapshot, ExitStack() as final_stack:
            if candidate_snapshot.sha256 != inspection.sqsh_sha256:
                raise ImageVerificationError("image inspection does not match the imported SQSH digest")
            final_snapshot: _VerifiedSqshSnapshot | None = None

            def publish_before_registry(_image: RegisteredImage) -> None:
                nonlocal created_artifact_identity, final_snapshot
                _verify_snapshot(image, candidate_snapshot)
                ensure_private_directory(artifact_directory, parents=False)
                created_artifact_identity = _publish_candidate(
                    candidate_path,
                    artifact_path,
                    expected_identity=candidate_snapshot.facts[:2],
                )
                try:
                    final_snapshot = final_stack.enter_context(_open_verified_sqsh(artifact_path))
                except ImageVerificationError as error:
                    if created_artifact_identity is None:
                        raise ImageConflictError(
                            f"image artifact path {artifact_path!s} cannot be reused as a verified SQSH artifact"
                        ) from error
                    raise
                if final_snapshot.sha256 != inspection.sqsh_sha256:
                    if created_artifact_identity is not None:
                        raise ImageVerificationError("published SQSH artifact does not match its inspection digest")
                    raise ImageConflictError(f"image artifact path {artifact_path!s} contains conflicting bytes")
                _sync_snapshot(final_snapshot)
                _verify_snapshot(image, final_snapshot)

            def verify_after_registry(_image: RegisteredImage) -> None:
                if final_snapshot is None:
                    raise ImageVerificationError("published SQSH artifact was not opened for verification")
                _verify_snapshot(image, final_snapshot)

            def rollback_after_failure(_image: RegisteredImage) -> None:
                if created_artifact_identity is not None:
                    _remove_artifact(artifact_path, expected_identity=created_artifact_identity)

            return self._registry.register(
                image,
                verify_before_publish=publish_before_registry,
                verify_after_publish=verify_after_registry,
                rollback_after_failure=rollback_after_failure,
                replace=replace,
            )

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
    with _open_verified_sqsh(path) as snapshot:
        return snapshot.sha256


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
    with _open_verified_sqsh(Path(image.path)) as snapshot:
        _verify_snapshot(image, snapshot)


@dataclass(frozen=True, slots=True)
class _VerifiedSqshSnapshot:
    path: Path
    descriptor: int
    facts: tuple[int, int, int, int]
    sha256: Sha256Digest

    def verify_unchanged(self) -> None:
        """Verify that the open bytes and their public path still match the snapshot."""
        descriptor_status = os.fstat(self.descriptor)
        path_status = self.path.lstat()
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or not stat.S_ISREG(path_status.st_mode)
            or _get_file_facts(descriptor_status) != self.facts
            or _get_file_facts(path_status) != self.facts
            or _hash_descriptor(self.descriptor) != self.sha256
        ):
            raise ImageVerificationError(f"image path {self.path} changed while it was being verified")


@contextmanager
def _open_verified_sqsh(path: Path) -> Iterator[_VerifiedSqshSnapshot]:
    descriptor: int | None = None
    try:
        before_open = path.lstat()
        if not stat.S_ISREG(before_open.st_mode):
            raise ImageVerificationError(f"image path {path} must be a regular non-symlink file")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise ImageVerificationError(f"image path {path} changed while it was being opened")
        facts = _get_file_facts(after_open)
        snapshot = _VerifiedSqshSnapshot(
            path=path,
            descriptor=descriptor,
            facts=facts,
            sha256=_hash_descriptor(descriptor),
        )
        snapshot.verify_unchanged()
        yield snapshot
    except FileNotFoundError as error:
        raise ImageVerificationError(f"image path {path} does not exist") from error
    except ImageVerificationError:
        raise
    except OSError as error:
        raise ImageVerificationError(f"cannot read image path {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _hash_descriptor(descriptor: int) -> Sha256Digest:
    digest = hashlib.sha256()
    offset = 0
    while block := os.pread(descriptor, _HASH_BLOCK_SIZE, offset):
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def _get_file_facts(status: os.stat_result) -> tuple[int, int, int, int]:
    # Publishing through a hard link changes ctime without changing the verified
    # inode or its bytes, so ctime cannot be part of the stable identity tuple.
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def _verify_snapshot(image: RegisteredImage, snapshot: _VerifiedSqshSnapshot) -> None:
    try:
        snapshot.verify_unchanged()
    except ImageVerificationError as error:
        raise ImageVerificationError(f"registered image {image.name!r} no longer matches its SQSH digest") from error
    if snapshot.sha256 != image.sqsh_sha256:
        raise ImageVerificationError(f"registered image {image.name!r} no longer matches its SQSH digest")


def _publish_candidate(
    candidate_path: Path,
    artifact_path: Path,
    *,
    expected_identity: tuple[int, int],
) -> tuple[int, int] | None:
    try:
        with (
            open_verified_directory(candidate_path.parent) as candidate_directory_descriptor,
            open_verified_directory(artifact_path.parent) as artifact_directory_descriptor,
        ):
            candidate_status = os.stat(
                candidate_path.name,
                dir_fd=candidate_directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(candidate_status.st_mode)
                or (
                    candidate_status.st_dev,
                    candidate_status.st_ino,
                )
                != expected_identity
            ):
                raise OSError(f"imported SQSH candidate {candidate_path!s} changed before publication")
            try:
                os.link(
                    candidate_path.name,
                    artifact_path.name,
                    src_dir_fd=candidate_directory_descriptor,
                    dst_dir_fd=artifact_directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                return None
            try:
                published_status = os.stat(
                    artifact_path.name,
                    dir_fd=artifact_directory_descriptor,
                    follow_symlinks=False,
                )
                published_identity = (published_status.st_dev, published_status.st_ino)
                if published_identity != expected_identity:
                    raise OSError(f"published SQSH artifact {artifact_path!s} has an unexpected identity")
                os.fsync(artifact_directory_descriptor)
                candidate_status = os.stat(
                    candidate_path.name,
                    dir_fd=candidate_directory_descriptor,
                    follow_symlinks=False,
                )
                if (candidate_status.st_dev, candidate_status.st_ino) != published_identity:
                    raise OSError(f"imported SQSH candidate {candidate_path!s} changed during publication")
                os.unlink(candidate_path.name, dir_fd=candidate_directory_descriptor)
                os.fsync(candidate_directory_descriptor)
            except OSError:
                _remove_created_artifact_from_directory(
                    artifact_directory_descriptor,
                    artifact_path.name,
                )
                raise
            return published_identity
    except OSError as error:
        raise ImageVerificationError(f"cannot publish image artifact {artifact_path!s}") from error


def _sync_snapshot(snapshot: _VerifiedSqshSnapshot) -> None:
    try:
        os.fsync(snapshot.descriptor)
    except OSError as error:
        raise ImageVerificationError(f"cannot synchronize image artifact {snapshot.path!s}") from error


def _remove_artifact(artifact_path: Path, *, expected_identity: tuple[int, int]) -> None:
    try:
        with open_verified_directory(artifact_path.parent) as artifact_directory_descriptor:
            _remove_artifact_from_directory(
                artifact_directory_descriptor,
                artifact_path.name,
                expected_identity=expected_identity,
            )
    except OSError:
        pass


def _remove_artifact_from_directory(
    artifact_directory_descriptor: int,
    artifact_name: str,
    *,
    expected_identity: tuple[int, int],
) -> None:
    try:
        artifact_status = os.stat(
            artifact_name,
            dir_fd=artifact_directory_descriptor,
            follow_symlinks=False,
        )
        if (artifact_status.st_dev, artifact_status.st_ino) != expected_identity:
            return
        os.unlink(artifact_name, dir_fd=artifact_directory_descriptor)
        os.fsync(artifact_directory_descriptor)
    except OSError:
        pass


def _remove_created_artifact_from_directory(
    artifact_directory_descriptor: int,
    artifact_name: str,
) -> None:
    try:
        os.unlink(artifact_name, dir_fd=artifact_directory_descriptor)
        os.fsync(artifact_directory_descriptor)
    except OSError:
        pass
