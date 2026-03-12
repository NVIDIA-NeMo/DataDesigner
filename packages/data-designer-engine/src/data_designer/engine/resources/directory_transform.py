# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, Generic, TypeVar, get_args, get_origin

from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from data_designer.config.seed_source import DirectoryListingTransform, DirectorySeedTransform
from data_designer.errors import DataDesignerError
from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry

TransformConfigT = TypeVar("TransformConfigT", bound=DirectorySeedTransform)


class DirectoryTransformError(DataDesignerError): ...


@dataclass(frozen=True)
class DirectoryTransformContext:
    """Context object passed to directory transforms.

    The filesystem is rooted at ``root_path`` so transform implementations can work
    with relative paths while deciding their own traversal and file access strategy.
    """

    fs: AbstractFileSystem
    root_path: Path


class DirectoryTransform(ABC, Generic[TransformConfigT]):
    def __init__(self, config: TransformConfigT) -> None:
        config_type = self.get_config_type()
        if isinstance(config, config_type):
            self._config = config
            return
        self._config = config_type.model_validate(config)

    @classmethod
    def get_config_type(cls) -> type[TransformConfigT]:
        for base in getattr(cls, "__orig_bases__", []):
            origin = get_origin(base)
            if origin is DirectoryTransform:
                args = get_args(base)
                if args:
                    config_type = args[0]
                    origin_type = get_origin(config_type) or config_type
                    if isinstance(origin_type, type) and issubclass(origin_type, DirectorySeedTransform):
                        return config_type
        raise DirectoryTransformError(
            f"Could not determine config type for {cls.__name__!r}. "
            "DirectoryTransform implementations must define a DirectorySeedTransform generic type argument."
        )

    @classmethod
    def get_transform_type(cls) -> str:
        config_type = cls.get_config_type()
        field = config_type.model_fields.get("transform_type")
        if field is None or not isinstance(field.default, str):
            raise DirectoryTransformError(
                f"Directory transform config {config_type.__name__!r} must define a string transform_type default."
            )
        return field.default

    @property
    def config(self) -> TransformConfigT:
        return self._config

    @abstractmethod
    def normalize(self, *, context: DirectoryTransformContext) -> list[dict[str, Any]]: ...


class DirectoryListingDirectoryTransform(DirectoryTransform[DirectoryListingTransform]):
    def normalize(self, *, context: DirectoryTransformContext) -> list[dict[str, Any]]:
        max_depth = None if self.config.recursive else 1
        relative_paths = [
            _normalize_relative_path(path) for path in context.fs.find("", withdirs=False, maxdepth=max_depth)
        ]
        matched_paths = [
            relative_path
            for relative_path in relative_paths
            if fnmatchcase(PurePosixPath(relative_path).name, self.config.file_pattern)
        ]
        matched_paths.sort()

        if not matched_paths:
            search_scope = "under" if self.config.recursive else "directly under"
            raise DirectoryTransformError(
                f"No files matched file_pattern {self.config.file_pattern!r} {search_scope} {context.root_path}"
            )

        return [
            {
                "source_kind": "directory_file",
                "source_path": str(context.root_path / relative_path),
                "relative_path": relative_path,
                "file_name": PurePosixPath(relative_path).name,
            }
            for relative_path in matched_paths
        ]


class DirectoryTransformRegistry:
    def __init__(self, transforms: Sequence[type[DirectoryTransform[Any]]]):
        self._transforms: dict[str, type[DirectoryTransform[Any]]] = {}
        for transform_type in transforms:
            self.add_transform(transform_type)

    def add_transform(self, transform_type: type[DirectoryTransform[Any]]) -> None:
        registered_name = transform_type.get_transform_type()
        if registered_name in self._transforms:
            raise DirectoryTransformError(f"A directory transform for {registered_name!r} already exists")
        self._transforms[registered_name] = transform_type

    def create_transform(self, config: DirectorySeedTransform) -> DirectoryTransform[Any]:
        try:
            transform_type = self._transforms[config.transform_type]
        except KeyError:
            raise DirectoryTransformError(f"No directory transform found for transform_type {config.transform_type!r}")
        return transform_type(config)


def create_default_directory_transform_registry() -> DirectoryTransformRegistry:
    transforms: list[type[DirectoryTransform[Any]]] = [DirectoryListingDirectoryTransform]
    for plugin in PluginRegistry().get_plugins(PluginType.DIRECTORY_TRANSFORM):
        transforms.append(plugin.impl_cls)
    return DirectoryTransformRegistry(transforms)


def create_directory_transform_context(root_path: Path) -> DirectoryTransformContext:
    resolved_root_path = root_path.resolve()
    rooted_fs = DirFileSystem(path=str(resolved_root_path), fs=LocalFileSystem())
    return DirectoryTransformContext(fs=rooted_fs, root_path=resolved_root_path)


def _normalize_relative_path(path: str) -> str:
    return path.lstrip("/")
