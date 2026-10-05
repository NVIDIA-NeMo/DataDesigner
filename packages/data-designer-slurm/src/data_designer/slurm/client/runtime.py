# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect and bind the native Python runtime used by Slurm client steps."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from data_designer.slurm.client.environment import inspect_distributions
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.contracts import compute_canonical_json_sha256, validate_absolute_path
from data_designer.slurm.planning import ResolvedClientRuntime

_REQUIRED_DISTRIBUTIONS = frozenset(
    {
        "aiohttp",
        "data-designer",
        "data-designer-config",
        "data-designer-engine",
        "data-designer-slurm",
        "pip",
    }
)
_PROXY_AIOHTTP_VERSIONS = SpecifierSet(">=3.14.3,<4")
_DATA_DESIGNER_DISTRIBUTIONS = frozenset(
    {"data-designer", "data-designer-config", "data-designer-engine", "data-designer-slurm"}
)


class ClientRuntimeInspectionError(RuntimeError):
    """Raised when the active CLI interpreter cannot be used on compute nodes."""


class ClientRuntimeInspector:
    """Resolve the active CLI interpreter into an immutable planning contract."""

    def inspect(self) -> ResolvedClientRuntime:
        """Return verified facts for the Python interpreter running the CLI."""
        try:
            python_executable = validate_absolute_path(Path(sys.executable).absolute().as_posix())
            if not os.access(python_executable, os.X_OK):
                raise ClientRuntimeInspectionError("client Python executable is unavailable")
            cache_tag = sys.implementation.cache_tag
            if cache_tag is None:
                raise ClientRuntimeInspectionError("client Python does not expose an ABI tag")
            python_abi = f"cp{cache_tag.removeprefix('cpython-')}" if cache_tag.startswith("cpython-") else cache_tag
            distributions = inspect_distributions(None)
            names = {distribution.name for distribution in distributions}
            missing = _REQUIRED_DISTRIBUTIONS.difference(names)
            if missing:
                raise ClientRuntimeInspectionError(
                    f"required client distributions are not installed: {', '.join(sorted(missing))}"
                )
            package_versions = {
                distribution.version
                for distribution in distributions
                if distribution.name in _DATA_DESIGNER_DISTRIBUTIONS
            }
            if len(package_versions) != 1:
                raise ClientRuntimeInspectionError("Data Designer packages in the client Python must share one version")
            aiohttp_version = next(
                distribution.version for distribution in distributions if distribution.name == "aiohttp"
            )
            try:
                compatible_aiohttp = Version(aiohttp_version) in _PROXY_AIOHTTP_VERSIONS
            except InvalidVersion:
                compatible_aiohttp = False
            if not compatible_aiohttp:
                raise ClientRuntimeInspectionError("client Python requires aiohttp>=3.14.3,<4 for the inference proxy")
            identity = {
                "python_executable": python_executable,
                "python_implementation": platform.python_implementation().casefold(),
                "python_version": platform.python_version(),
                "python_abi": python_abi,
                "distributions": distributions,
            }
            fingerprint_payload = identity | {
                "distributions": [distribution.model_dump(mode="json") for distribution in distributions]
            }
            return ResolvedClientRuntime(
                **identity,
                runtime_sha256=compute_canonical_json_sha256(fingerprint_payload),
            )
        except ClientRuntimeInspectionError:
            raise
        except ClientWorkerError as error:
            raise ClientRuntimeInspectionError(error.redacted_message) from None
        except (OSError, ValueError):
            raise ClientRuntimeInspectionError("client Python runtime cannot be inspected") from None


__all__ = ["ClientRuntimeInspectionError", "ClientRuntimeInspector"]
