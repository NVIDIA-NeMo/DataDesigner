# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic workspace-derived persistence for immutable Slurm image facts."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from data_designer.slurm.config import ImageRef
from data_designer.slurm.contracts import Identifier, validate_absolute_path
from data_designer.slurm.images.errors import (
    ImageConflictError,
    ImageNotFoundError,
    ImageRegistryError,
)
from data_designer.slurm.images.records import ImageRegistryDocument, RegisteredImage

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_LOCK_DIRECTORY_NAME = ".locks"
_REGISTRY_FILENAME = "registry.yaml"
_REGISTRY_LOCK_FILENAME = "registry.lock"
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class ImageRegistryStore:
    """Persist image aliases beneath one explicitly selected shared workspace."""

    def __init__(self, workspace_root: str | Path) -> None:
        try:
            normalized_root = validate_absolute_path(Path(workspace_root).as_posix())
        except ValueError as error:
            raise ImageRegistryError(f"invalid image workspace root {workspace_root!s}") from error
        self._image_root = Path(normalized_root) / "images"
        self._registry_path = self._image_root / _REGISTRY_FILENAME
        self._lock_directory = self._image_root / _LOCK_DIRECTORY_NAME

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
        name = _validate_image_alias(name)
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

    def register(
        self,
        image: RegisteredImage,
        *,
        verify_before_publish: Callable[[RegisteredImage], None],
        replace: bool = False,
    ) -> RegisteredImage:
        """Atomically add or explicitly replace one verified image alias."""
        self._ensure_storage()
        alias_lock_name = f"alias-{image.name}.lock"
        with self._open_storage() as (image_root_descriptor, lock_directory_descriptor):
            with (
                _acquire_file_lock(
                    lock_directory_descriptor,
                    alias_lock_name,
                    self._lock_directory / alias_lock_name,
                ),
                _acquire_file_lock(
                    lock_directory_descriptor,
                    _REGISTRY_LOCK_FILENAME,
                    self._lock_directory / _REGISTRY_LOCK_FILENAME,
                ),
            ):
                snapshot = self._load_from_directory(image_root_descriptor)
                existing = next((entry for entry in snapshot.images if entry.name == image.name), None)
                if existing is not None and not replace:
                    raise ImageConflictError(f"image alias {image.name!r} is already registered")
                self._validate_path_facts(snapshot, image, replacing=existing)
                images = tuple(entry for entry in snapshot.images if entry.name != image.name) + (image,)
                updated = ImageRegistryDocument(
                    schema_version=1,
                    images=tuple(sorted(images, key=lambda entry: entry.name)),
                )
                self._save(
                    image_root_descriptor,
                    updated,
                    verify_before_publish=lambda: verify_before_publish(image),
                )
        return image

    def unregister(self, name: Identifier) -> RegisteredImage:
        """Atomically remove an alias without deleting its underlying SQSH artifact."""
        name = _validate_image_alias(name)
        self._ensure_storage()
        alias_lock_name = f"alias-{name}.lock"
        with self._open_storage() as (image_root_descriptor, lock_directory_descriptor):
            with (
                _acquire_file_lock(
                    lock_directory_descriptor,
                    alias_lock_name,
                    self._lock_directory / alias_lock_name,
                ),
                _acquire_file_lock(
                    lock_directory_descriptor,
                    _REGISTRY_LOCK_FILENAME,
                    self._lock_directory / _REGISTRY_LOCK_FILENAME,
                ),
            ):
                snapshot = self._load_from_directory(image_root_descriptor)
                try:
                    removed = next(image for image in snapshot.images if image.name == name)
                except StopIteration:
                    raise ImageNotFoundError(f"image alias {name!r} is not registered") from None
                updated = ImageRegistryDocument(
                    schema_version=1,
                    images=tuple(image for image in snapshot.images if image.name != name),
                )
                self._save(image_root_descriptor, updated)
        return removed

    def _load(self) -> ImageRegistryDocument:
        try:
            with _open_verified_directory(self._image_root) as image_root_descriptor:
                return self._load_from_directory(image_root_descriptor)
        except FileNotFoundError:
            return ImageRegistryDocument(schema_version=1)
        except OSError as error:
            raise ImageRegistryError(f"cannot load image registry {self._registry_path}") from error

    def _load_from_directory(self, image_root_descriptor: int) -> ImageRegistryDocument:
        try:
            payload = yaml.load(
                _read_regular_text(image_root_descriptor, _REGISTRY_FILENAME, self._registry_path),
                Loader=_UniqueKeySafeLoader,
            )
            return ImageRegistryDocument.model_validate_json(json.dumps(payload))
        except FileNotFoundError:
            return ImageRegistryDocument(schema_version=1)
        except (OSError, RecursionError, TypeError, UnicodeError, ValueError, ValidationError, yaml.YAMLError) as error:
            raise ImageRegistryError(f"cannot load image registry {self._registry_path}") from error

    def _save(
        self,
        image_root_descriptor: int,
        snapshot: ImageRegistryDocument,
        *,
        verify_before_publish: Callable[[], None] | None = None,
    ) -> None:
        temporary_name: str | None = None
        descriptor: int | None = None
        try:
            descriptor, temporary_name = _create_temporary_file(image_root_descriptor)
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
            if verify_before_publish is not None:
                verify_before_publish()
            os.replace(
                temporary_name,
                _REGISTRY_FILENAME,
                src_dir_fd=image_root_descriptor,
                dst_dir_fd=image_root_descriptor,
            )
            os.fsync(image_root_descriptor)
        except (OSError, TypeError, yaml.YAMLError) as error:
            raise ImageRegistryError(f"cannot persist image registry {self._registry_path}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=image_root_descriptor)
                except OSError:
                    pass

    def _ensure_storage(self) -> None:
        try:
            _ensure_private_directory(self._image_root, parents=True)
            _ensure_private_directory(self._lock_directory, parents=False)
        except OSError as error:
            raise ImageRegistryError(f"cannot initialize image registry beneath {self._image_root}") from error

    @contextmanager
    def _open_storage(self) -> Iterator[tuple[int, int]]:
        try:
            with _open_verified_directory(self._image_root) as image_root_descriptor:
                with _open_verified_child_directory(
                    image_root_descriptor,
                    _LOCK_DIRECTORY_NAME,
                    self._lock_directory,
                ) as lock_directory_descriptor:
                    yield image_root_descriptor, lock_directory_descriptor
        except OSError as error:
            raise ImageRegistryError(f"cannot access image registry beneath {self._image_root}") from error

    @staticmethod
    def _validate_path_facts(
        snapshot: ImageRegistryDocument,
        image: RegisteredImage,
        *,
        replacing: RegisteredImage | None,
    ) -> None:
        for existing in snapshot.images:
            if replacing is not None and existing.name == replacing.name:
                continue
            if existing.path != image.path:
                continue
            if existing.immutable_facts != image.immutable_facts:
                raise ImageConflictError(f"image path {image.path!r} already has different immutable facts")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@contextmanager
def _acquire_file_lock(directory_descriptor: int, name: str, display_path: Path) -> Iterator[None]:
    """Acquire one exclusive advisory file lock for a registry mutation."""
    descriptor: int | None = None
    try:
        # macOS can report ENOENT to the losing openat(O_CREAT) caller when two
        # processes create the same lock file concurrently. Retrying then opens
        # the winner's regular file with the same no-follow boundary.
        for _ in range(10):
            try:
                descriptor = os.open(
                    name,
                    os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    _FILE_MODE,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                continue
            break
        if descriptor is None:
            raise OSError(f"lock path {display_path} could not be opened")
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"lock path {display_path} is not a regular file")
        os.fchmod(descriptor, _FILE_MODE)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ImageRegistryError(f"cannot lock image registry target {display_path}") from error

    try:
        yield
    finally:
        assert descriptor is not None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_image_alias(name: str) -> Identifier:
    """Validate an image alias before deriving any filesystem path from it."""
    try:
        return _IDENTIFIER_ADAPTER.validate_python(name, strict=True)
    except ValidationError as error:
        raise ImageRegistryError(f"invalid image alias {name!r}") from error


def _ensure_private_directory(path: Path, *, parents: bool) -> None:
    path.mkdir(mode=_DIRECTORY_MODE, parents=parents, exist_ok=True)
    with _open_verified_directory(path) as descriptor:
        os.fchmod(descriptor, _DIRECTORY_MODE)


@contextmanager
def _open_verified_directory(path: Path) -> Iterator[int]:
    descriptor: int | None = None
    try:
        before_open = path.lstat()
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"registry directory {path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"registry directory {path} changed while it was being opened")
        if not stat.S_ISDIR(after_open.st_mode):
            raise OSError(f"registry directory {path} is not a directory")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _open_verified_child_directory(
    parent_descriptor: int,
    name: str,
    display_path: Path,
) -> Iterator[int]:
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"registry directory {display_path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"registry directory {display_path} changed while it was being opened")
        if not stat.S_ISDIR(after_open.st_mode):
            raise OSError(f"registry directory {display_path} is not a directory")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _create_temporary_file(directory_descriptor: int) -> tuple[int, str]:
    for _ in range(100):
        name = f".registry.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                _FILE_MODE,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        return descriptor, name
    raise OSError("cannot allocate a unique registry snapshot name")


def _read_regular_text(directory_descriptor: int, name: str, display_path: Path) -> str:
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before_open.st_mode):
            raise OSError(f"registry path {display_path} is not a regular file")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"registry path {display_path} changed while it was being opened")
        registry_file = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = None
        with registry_file:
            return registry_file.read()
    finally:
        if descriptor is not None:
            os.close(descriptor)
