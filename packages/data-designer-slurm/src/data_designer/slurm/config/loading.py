# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict local loading and cluster-profile selection."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar, cast

import yaml
from pydantic import ValidationError
from yaml.nodes import MappingNode

from data_designer.slurm.config.profiles import (
    SelectedSlurmProfile,
    SlurmProfile,
    SlurmProfileCatalog,
    injected_profile,
    select_profile,
)
from data_designer.slurm.config.run import DataDesignerSlurmConfig

PROFILE_FILE_ENVIRONMENT = "DATA_DESIGNER_SLURM_PROFILE_FILE"
DEFAULT_PROFILE_FILE_NAME = ".data-designer-slurm-profile.yml"

_Config = TypeVar("_Config", DataDesignerSlurmConfig, SlurmProfileCatalog)
_HostnameResolver = Callable[[], tuple[str, ...]]


class ConfigLoadError(ValueError):
    """Raised when a local Slurm configuration file is not strict and valid."""


class _StrictYamlLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _StrictYamlLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ConfigLoadError("YAML merge keys are not supported")
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ConfigLoadError("configuration mapping keys must be scalar values") from error
        if duplicate:
            raise ConfigLoadError(f"duplicate configuration key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_run_config(path: str | Path) -> DataDesignerSlurmConfig:
    """Load one strict local YAML or JSON run declaration."""
    return _load_config(path, DataDesignerSlurmConfig)


def load_profile_catalog(path: str | Path) -> SlurmProfileCatalog:
    """Load one strict local YAML or JSON cluster-profile catalog."""
    return _load_config(path, SlurmProfileCatalog)


def resolve_profile(
    *,
    profile: SlurmProfile | None = None,
    catalog: SlurmProfileCatalog | None = None,
    profile_file: str | Path | None = None,
    cluster: str | None = None,
    hostnames: tuple[str, ...] | None = None,
    hostname_resolver: _HostnameResolver | None = None,
    environ: Mapping[str, str] | None = None,
    home_directory: str | Path | None = None,
) -> SelectedSlurmProfile:
    """Resolve an injected profile or select one catalog entry."""
    sources = sum(source is not None for source in (profile, catalog, profile_file))
    if sources > 1:
        raise ConfigLoadError("profile, catalog, and profile_file are mutually exclusive")
    if profile is not None:
        if cluster is not None:
            raise ConfigLoadError("an injected profile cannot be combined with cluster selection")
        return injected_profile(profile)

    catalog_path: str | None = None
    if catalog is None:
        path = _resolve_profile_path(
            profile_file,
            environ=os.environ if environ is None else environ,
            home_directory=home_directory,
        )
        catalog = load_profile_catalog(path)
        catalog_path = path.as_posix()

    if hostnames is None:
        resolver = hostname_resolver or _local_hostnames
        hostnames = resolver()
    normalized_hostnames = tuple(dict.fromkeys(hostname.strip().casefold() for hostname in hostnames if hostname))
    return select_profile(
        catalog,
        cluster=cluster,
        hostnames=normalized_hostnames,
        catalog_path=catalog_path,
    )


def _load_config(path: str | Path, config_type: type[_Config]) -> _Config:
    resolved_path = _normalize_file_path(path)
    try:
        contents = resolved_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigLoadError(f"cannot read configuration file {resolved_path}") from error
    try:
        payload = _parse_mapping(contents, suffix=resolved_path.suffix)
        _reject_environment_interpolation(payload)
        return config_type.model_validate(payload)
    except ConfigLoadError:
        raise
    except (ValidationError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ConfigLoadError(f"invalid configuration file {resolved_path}: {error}") from error


def _parse_mapping(contents: str, *, suffix: str) -> dict[str, object]:
    if suffix == ".json":
        payload = json.loads(contents, object_pairs_hook=_unique_json_object)
    else:
        events = yaml.parse(contents, Loader=yaml.SafeLoader)
        if any(getattr(event, "anchor", None) is not None for event in events):
            raise ConfigLoadError("YAML anchors and aliases are not supported")
        payload = yaml.load(contents, Loader=_StrictYamlLoader)
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ConfigLoadError("configuration root must be an object with string keys")
    return cast(dict[str, object], payload)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigLoadError(f"duplicate configuration key: {key!r}")
        result[key] = value
    return result


def _reject_environment_interpolation(value: object) -> None:
    if isinstance(value, str) and "${" in value:
        raise ConfigLoadError("environment interpolation is not supported")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_environment_interpolation(key)
            _reject_environment_interpolation(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_environment_interpolation(item)


def _resolve_profile_path(
    explicit_path: str | Path | None,
    *,
    environ: Mapping[str, str],
    home_directory: str | Path | None,
) -> Path:
    source = explicit_path
    if source is None:
        environment_path = environ.get(PROFILE_FILE_ENVIRONMENT)
        if environment_path is not None and not environment_path:
            raise ConfigLoadError(f"{PROFILE_FILE_ENVIRONMENT} must not be empty")
        source = environment_path
    if source is None:
        home = Path.home() if home_directory is None else Path(home_directory)
        source = home / DEFAULT_PROFILE_FILE_NAME
    return _normalize_file_path(source)


def _normalize_file_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix not in {".json", ".yaml", ".yml"}:
        raise ConfigLoadError("configuration path must end in .json, .yaml, or .yml")
    return resolved


def _local_hostnames() -> tuple[str, ...]:
    return socket.gethostname(), socket.getfqdn()
