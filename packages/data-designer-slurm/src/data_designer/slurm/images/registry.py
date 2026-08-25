# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic workspace-derived persistence for immutable Slurm image facts."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.config import ImageRef
from data_designer.slurm.contracts import Identifier, validate_absolute_path
from data_designer.slurm.images.errors import ImageConflictError, ImageNotFoundError, ImageRegistryError
from data_designer.slurm.images.records import ImageRegistrySnapshot, RegisteredImage

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class ImageRegistry:
    """Persist image aliases beneath one explicitly selected shared workspace."""

    def __init__(self, workspace_root: str | Path) -> None:
        try:
            normalized_root = validate_absolute_path(Path(workspace_root).as_posix())
        except ValueError as error:
            raise ImageRegistryError(f"invalid image workspace root {workspace_root!s}") from error
        self._image_root = Path(normalized_root) / "images"
        self._registry_path = self._image_root / "registry.yaml"
        self._lock_directory = self._image_root / ".locks"

    @property
    def image_root(self) -> Path:
        """Return the workspace-derived image root."""
        return self._image_root

    @property
    def registry_path(self) -> Path:
        """Return the workspace-derived registry record path."""
        return self._registry_path

    def list_images(self) -> tuple[RegisteredImage, ...]:
        """Return all registered aliases in deterministic order."""
        return self._load().images

    def get_by_name(self, name: Identifier) -> RegisteredImage:
        """Return one registered alias."""
        name = validate_image_alias(name)
        for image in self._load().images:
            if image.name == name:
                return image
        raise ImageNotFoundError(f"image alias {name!r} is not registered")

    def get_by_path(self, path: str | Path) -> RegisteredImage:
        """Return immutable facts for one registered absolute SQSH path."""
        try:
            normalized_path = ImageRef(path=Path(path).as_posix()).path
        except ValidationError as error:
            raise ImageRegistryError(f"invalid registered image path {path!s}") from error
        assert normalized_path is not None
        matches = tuple(image for image in self._load().images if image.path == normalized_path)
        if not matches:
            raise ImageNotFoundError(f"image path {normalized_path!r} is not registered")
        return matches[0]

    def register(self, image: RegisteredImage, *, replace: bool = False) -> RegisteredImage:
        """Atomically add or explicitly replace one image alias."""
        self._ensure_storage()
        with (
            acquire_file_lock(self._get_alias_lock_path(image.name)),
            acquire_file_lock(self._get_registry_lock_path()),
        ):
            snapshot = self._load()
            existing = next((entry for entry in snapshot.images if entry.name == image.name), None)
            if existing is not None and not replace:
                raise ImageConflictError(f"image alias {image.name!r} is already registered")
            self._validate_path_facts(snapshot, image, replacing=existing)
            images = tuple(entry for entry in snapshot.images if entry.name != image.name) + (image,)
            updated = ImageRegistrySnapshot(images=tuple(sorted(images, key=lambda entry: entry.name)))
            self._save(updated)
        return image

    def unregister(self, name: Identifier) -> RegisteredImage:
        """Atomically remove an alias without deleting its underlying SQSH artifact."""
        name = validate_image_alias(name)
        self._ensure_storage()
        with acquire_file_lock(self._get_alias_lock_path(name)), acquire_file_lock(self._get_registry_lock_path()):
            snapshot = self._load()
            try:
                removed = next(image for image in snapshot.images if image.name == name)
            except StopIteration:
                raise ImageNotFoundError(f"image alias {name!r} is not registered") from None
            updated = ImageRegistrySnapshot(images=tuple(image for image in snapshot.images if image.name != name))
            self._save(updated)
        return removed

    def _load(self) -> ImageRegistrySnapshot:
        if not self._registry_path.exists():
            return ImageRegistrySnapshot()
        try:
            payload = yaml.safe_load(_read_regular_text(self._registry_path))
            return ImageRegistrySnapshot.model_validate_json(json.dumps(payload))
        except (OSError, TypeError, UnicodeError, ValidationError, yaml.YAMLError) as error:
            raise ImageRegistryError(f"cannot load image registry {self._registry_path}") from error

    def _save(self, snapshot: ImageRegistrySnapshot) -> None:
        temporary_path: Path | None = None
        descriptor: int | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=self._image_root,
                prefix=".registry.",
                suffix=".tmp",
                text=True,
            )
            temporary_path = Path(raw_path)
            os.fchmod(descriptor, _FILE_MODE)
            output = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = None
            with output:
                yaml.safe_dump(
                    snapshot.model_dump(mode="json"),
                    output,
                    default_flow_style=False,
                    sort_keys=True,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self._registry_path)
            _sync_directory(self._image_root)
        except (OSError, TypeError, yaml.YAMLError) as error:
            raise ImageRegistryError(f"cannot persist image registry {self._registry_path}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _ensure_storage(self) -> None:
        try:
            self._image_root.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
            self._lock_directory.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
        except OSError as error:
            raise ImageRegistryError(f"cannot initialize image registry beneath {self._image_root}") from error

    def _get_alias_lock_path(self, name: Identifier) -> Path:
        return self._lock_directory / f"alias-{name}.lock"

    def _get_registry_lock_path(self) -> Path:
        return self._lock_directory / "registry.lock"

    @staticmethod
    def _validate_path_facts(
        snapshot: ImageRegistrySnapshot,
        image: RegisteredImage,
        *,
        replacing: RegisteredImage | None,
    ) -> None:
        for existing in snapshot.images:
            if replacing is not None and existing.name == replacing.name:
                continue
            if existing.path != image.path:
                continue
            existing_facts = (existing.sqsh_sha256, existing.source_oci_digest, existing.inspection)
            new_facts = (image.sqsh_sha256, image.source_oci_digest, image.inspection)
            if existing_facts != new_facts:
                raise ImageConflictError(f"image path {image.path!r} already has different immutable facts")


@contextmanager
def acquire_file_lock(path: Path) -> Iterator[None]:
    """Acquire one exclusive advisory file lock for a registry mutation."""
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), _FILE_MODE)
        os.fchmod(descriptor, _FILE_MODE)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ImageRegistryError(f"cannot lock image registry target {path}") from error

    try:
        yield
    finally:
        assert descriptor is not None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def validate_image_alias(name: str) -> Identifier:
    """Validate an image alias before deriving any filesystem path from it."""
    try:
        return _IDENTIFIER_ADAPTER.validate_python(name, strict=True)
    except ValidationError as error:
        raise ImageRegistryError(f"invalid image alias {name!r}") from error


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_text(path: Path) -> str:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        registry_file = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = None
        with registry_file:
            if not stat.S_ISREG(os.fstat(registry_file.fileno()).st_mode):
                raise OSError(f"registry path {path} is not a regular file")
            return registry_file.read()
    finally:
        if descriptor is not None:
            os.close(descriptor)
