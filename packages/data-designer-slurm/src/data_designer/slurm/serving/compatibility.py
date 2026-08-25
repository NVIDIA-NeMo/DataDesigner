# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-owned compatibility rules for inspected serving runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from packaging.version import InvalidVersion, Version
from pydantic import StringConstraints, model_validator

from data_designer.slurm.contracts import ContractValue, Identifier

RuntimeVersion = Annotated[str, StringConstraints(min_length=1, max_length=128)]


@dataclass(frozen=True, slots=True)
class _VllmCapabilities:
    supports_single_node: bool
    supports_multi_node: bool
    supports_http_readiness: bool
    supports_queue_backpressure: bool
    supports_coordinated_failure: bool


_V1_CAPABILITIES = _VllmCapabilities(
    supports_single_node=True,
    supports_multi_node=True,
    supports_http_readiness=True,
    supports_queue_backpressure=True,
    supports_coordinated_failure=True,
)
_SUPPORTED_VLLM_CAPABILITIES = {
    (0, 21): _V1_CAPABILITIES,
    (0, 22): _V1_CAPABILITIES,
    (0, 23): _V1_CAPABILITIES,
}


class UnsupportedServingRuntimeError(ValueError):
    """Raised when an inspected serving runtime lacks a tested V1 contract."""


class VllmRuntimeCompatibility(ContractValue):
    """Package-owned capabilities for one inspected vLLM release series."""

    runtime_version: RuntimeVersion
    runtime_series: Identifier
    supports_single_node: bool
    supports_multi_node: bool
    supports_http_readiness: bool
    supports_queue_backpressure: bool
    supports_coordinated_failure: bool

    @model_validator(mode="after")
    def validate_runtime_series(self) -> VllmRuntimeCompatibility:
        version, expected_capabilities = _resolve_supported_version(self.runtime_version)
        expected_series = ".".join(str(component) for component in version.release[:2])
        if self.runtime_series != expected_series:
            raise ValueError("compatibility runtime series must match its version")
        actual_capabilities = _VllmCapabilities(
            supports_single_node=self.supports_single_node,
            supports_multi_node=self.supports_multi_node,
            supports_http_readiness=self.supports_http_readiness,
            supports_queue_backpressure=self.supports_queue_backpressure,
            supports_coordinated_failure=self.supports_coordinated_failure,
        )
        if actual_capabilities != expected_capabilities:
            raise ValueError("compatibility capabilities must match the package-owned runtime mapping")
        return self


def resolve_vllm_compatibility(runtime_version: str) -> VllmRuntimeCompatibility:
    """Map an inspected vLLM version to the package's tested V1 behavior."""
    version, capabilities = _resolve_supported_version(runtime_version)
    return VllmRuntimeCompatibility(
        runtime_version=runtime_version,
        runtime_series=".".join(str(component) for component in version.release[:2]),
        supports_single_node=capabilities.supports_single_node,
        supports_multi_node=capabilities.supports_multi_node,
        supports_http_readiness=capabilities.supports_http_readiness,
        supports_queue_backpressure=capabilities.supports_queue_backpressure,
        supports_coordinated_failure=capabilities.supports_coordinated_failure,
    )


def _resolve_supported_version(runtime_version: str) -> tuple[Version, _VllmCapabilities]:
    try:
        version = Version(runtime_version)
    except InvalidVersion as error:
        raise UnsupportedServingRuntimeError(f"invalid inspected vLLM version: {runtime_version!r}") from error
    series = version.release[:2]
    if version.epoch or version.is_prerelease or version.is_devrelease or series not in _SUPPORTED_VLLM_CAPABILITIES:
        raise UnsupportedServingRuntimeError(f"unsupported inspected vLLM version: {runtime_version!r}")
    return version, _SUPPORTED_VLLM_CAPABILITIES[series]
