# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-owned compatibility rules for inspected serving runtimes."""

from __future__ import annotations

from typing import Annotated, Literal

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import StringConstraints, model_validator

from data_designer.slurm.contracts import ContractValue, Identifier, validate_plain_text

_MAX_RUNTIME_VERSION_LENGTH = 128
RuntimeVersion = Annotated[str, StringConstraints(min_length=1, max_length=_MAX_RUNTIME_VERSION_LENGTH)]


_TESTED_VLLM_VERSION_RANGE = SpecifierSet(">=0.21,<0.23")


class UnsupportedServingRuntimeError(ValueError):
    """Raised when an inspected serving runtime lacks a tested V1 contract."""


class VllmRuntimeCompatibility(ContractValue):
    """Inspected vLLM identity accepted by one package contract revision."""

    runtime_version: RuntimeVersion
    runtime_series: Identifier
    contract_version: Literal["v1"] = "v1"

    @model_validator(mode="after")
    def validate_runtime_series(self) -> VllmRuntimeCompatibility:
        version = _resolve_supported_version(self.runtime_version)
        expected_series = ".".join(str(component) for component in version.release[:2])
        if self.runtime_series != expected_series:
            raise ValueError("compatibility runtime series must match its version")
        return self


def resolve_vllm_compatibility(runtime_version: str) -> VllmRuntimeCompatibility:
    """Map an inspected vLLM version to the package's tested V1 behavior."""
    version = _resolve_supported_version(runtime_version)
    return VllmRuntimeCompatibility(
        runtime_version=runtime_version,
        runtime_series=".".join(str(component) for component in version.release[:2]),
    )


def _resolve_supported_version(runtime_version: str) -> Version:
    try:
        validate_plain_text(runtime_version, field_name="vLLM version")
        if len(runtime_version) > _MAX_RUNTIME_VERSION_LENGTH:
            raise ValueError("vLLM version is too long")
        version = Version(runtime_version)
    except (InvalidVersion, ValueError) as error:
        raise UnsupportedServingRuntimeError(f"invalid inspected vLLM version: {runtime_version!r}") from error
    if runtime_version != str(version):
        raise UnsupportedServingRuntimeError(f"noncanonical inspected vLLM version: {runtime_version!r}")
    if (
        version.epoch
        or version.is_prerelease
        or version.is_devrelease
        or not _TESTED_VLLM_VERSION_RANGE.contains(version, prereleases=False)
    ):
        raise UnsupportedServingRuntimeError(f"unsupported inspected vLLM version: {runtime_version!r}")
    return version
