# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic image-environment facts for package inspector tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from data_designer.slurm.config import InstalledDistribution
from data_designer.slurm.images.errors import ImageInspectionError


@dataclass(frozen=True, slots=True)
class FakeInspectionEnvironment:
    """Expose explicit client and serving facts without entering a container."""

    python_implementation: str = "cpython"
    python_version: str = "3.12.12"
    python_abi: str = "cp312"
    distributions: tuple[InstalledDistribution, ...] = ()
    distribution_versions: Mapping[str, str] = field(default_factory=dict)
    executables: Mapping[str, str] = field(default_factory=dict)

    def get_python_implementation(self) -> str:
        """Return the configured Python implementation."""
        return self.python_implementation

    def get_python_version(self) -> str:
        """Return the configured Python version."""
        return self.python_version

    def get_python_abi(self) -> str:
        """Return the configured Python ABI."""
        return self.python_abi

    def list_distributions(self) -> tuple[InstalledDistribution, ...]:
        """Return the configured distribution inventory."""
        return self.distributions

    def get_distribution_console_script(self, name: str) -> tuple[str, str]:
        """Return one configured distribution version and console-script path."""
        try:
            return (self.distribution_versions[name], self.executables[name])
        except KeyError:
            raise ImageInspectionError(f"required distribution console script {name!r} is not installed") from None
