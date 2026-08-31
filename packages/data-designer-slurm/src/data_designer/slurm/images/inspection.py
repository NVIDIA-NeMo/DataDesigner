# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-owned factual inspectors executed inside target image environments."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Protocol

from packaging.utils import canonicalize_name
from pydantic import ValidationError

from data_designer.slurm.config import (
    ClientImageInspection,
    ImageInspectionRecord,
    InstalledDistribution,
    ServingImageInspection,
)
from data_designer.slurm.contracts import Identifier, Sha256Digest
from data_designer.slurm.images.errors import ImageInspectionError
from data_designer.slurm.images.resources.inspect_image import find_distribution_console_script

INSPECTOR_VERSION: Identifier = "inspector-1"
_REQUIRED_CLIENT_DISTRIBUTIONS = (
    "data-designer",
    "data-designer-config",
    "data-designer-engine",
    "data-designer-slurm",
    "pip",
)


class InspectionEnvironment(Protocol):
    """Minimal environment facts required by the package image inspectors."""

    def get_python_implementation(self) -> str:
        """Return the active Python implementation name."""

    def get_python_version(self) -> str:
        """Return the active Python semantic version."""

    def get_python_abi(self) -> str:
        """Return the active Python wheel ABI tag."""

    def list_distributions(self) -> tuple[InstalledDistribution, ...]:
        """Return the installed Python distribution inventory."""

    def get_distribution_console_script(self, name: str) -> tuple[str, str]:
        """Return one distribution version and its owned console-script path."""


class SystemInspectionEnvironment:
    """Read factual metadata from the environment in which the inspector runs."""

    def get_python_implementation(self) -> str:
        """Return the normalized active Python implementation."""
        return platform.python_implementation().casefold()

    def get_python_version(self) -> str:
        """Return the active Python version."""
        return platform.python_version()

    def get_python_abi(self) -> str:
        """Return the active Python wheel ABI tag."""
        cache_tag = sys.implementation.cache_tag
        if cache_tag is None:
            raise ImageInspectionError("active Python does not expose an ABI cache tag")
        if cache_tag.startswith("cpython-"):
            return f"cp{cache_tag.removeprefix('cpython-')}"
        return cache_tag

    def list_distributions(self) -> tuple[InstalledDistribution, ...]:
        """Return a normalized, deterministic distribution inventory."""
        versions_by_name: dict[str, str] = {}
        for distribution in importlib.metadata.distributions():
            raw_name = distribution.metadata.get("Name")
            if not raw_name:
                raise ImageInspectionError("installed distribution is missing its canonical name")
            name = canonicalize_name(raw_name)
            existing_version = versions_by_name.setdefault(name, distribution.version)
            if existing_version != distribution.version:
                raise ImageInspectionError(f"installed distribution {name!r} has conflicting versions")
        try:
            return tuple(
                InstalledDistribution(name=name, version=version) for name, version in sorted(versions_by_name.items())
            )
        except ValidationError as error:
            raise ImageInspectionError("installed distribution inventory is invalid") from error

    def get_distribution_console_script(self, name: str) -> tuple[str, str]:
        """Return one distribution version and its owned console-script path."""
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ImageInspectionError(f"required distribution {name!r} is not installed") from error
        try:
            executable = find_distribution_console_script(distribution, name)
        except (OSError, RuntimeError) as error:
            raise ImageInspectionError(str(error)) from error
        return (distribution.version, executable)


class ClientImageInspector:
    """Inspect the Python and dependency-installation facts of a client image."""

    def __init__(self, environment: InspectionEnvironment | None = None) -> None:
        self._environment = environment if environment is not None else SystemInspectionEnvironment()

    def inspect(self, sqsh_sha256: Sha256Digest) -> ImageInspectionRecord:
        """Return a digest-bound client inspection for the active image environment."""
        environment = self._environment
        try:
            distributions = tuple(sorted(environment.list_distributions(), key=lambda item: item.name))
            versions_by_name = {distribution.name: distribution.version for distribution in distributions}
            missing_distributions = tuple(
                name for name in _REQUIRED_CLIENT_DISTRIBUTIONS if name not in versions_by_name
            )
            if missing_distributions:
                missing = ", ".join(repr(name) for name in missing_distributions)
                raise ImageInspectionError(f"required client distributions are not installed: {missing}")
            installer_version, installer_path = environment.get_distribution_console_script("pip")
            if installer_version != versions_by_name["pip"]:
                raise ImageInspectionError("pip console script does not match the distribution inventory")
            inspection = ClientImageInspection(
                kind="client",
                python_implementation=environment.get_python_implementation(),
                python_version=environment.get_python_version(),
                python_abi=environment.get_python_abi(),
                distributions=distributions,
                installer_path=installer_path,
                installer_version=installer_version,
            )
            return ImageInspectionRecord(
                schema_version=1,
                inspector_version=INSPECTOR_VERSION,
                sqsh_sha256=sqsh_sha256,
                inspection=inspection,
            )
        except ImageInspectionError:
            raise
        except (OSError, ValueError) as error:
            raise ImageInspectionError("client image inspection produced invalid facts") from error


class ServingImageInspector:
    """Inspect package-owned vLLM runtime facts in a serving image."""

    def __init__(self, environment: InspectionEnvironment | None = None) -> None:
        self._environment = environment if environment is not None else SystemInspectionEnvironment()

    def inspect(self, sqsh_sha256: Sha256Digest) -> ImageInspectionRecord:
        """Return a digest-bound serving inspection for the active image environment."""
        environment = self._environment
        try:
            runtime_version, executable_path = environment.get_distribution_console_script("vllm")
            inspection = ServingImageInspection(
                kind="serving",
                server_type="vllm",
                runtime_version=runtime_version,
                executable_path=executable_path,
            )
            return ImageInspectionRecord(
                schema_version=1,
                inspector_version=INSPECTOR_VERSION,
                sqsh_sha256=sqsh_sha256,
                inspection=inspection,
            )
        except ImageInspectionError:
            raise
        except (OSError, ValueError) as error:
            raise ImageInspectionError("serving image inspection produced invalid facts") from error
