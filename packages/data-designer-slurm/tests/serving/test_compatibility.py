# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.slurm.serving import UnsupportedServingRuntimeError, resolve_vllm_compatibility


@pytest.mark.parametrize("runtime_version", ["0.21.0", "0.22.4", "0.23.0+container.1"])
def test_supported_vllm_release_series_have_complete_v1_capabilities(runtime_version: str) -> None:
    compatibility = resolve_vllm_compatibility(runtime_version)

    assert compatibility.runtime_series in {"0.21", "0.22", "0.23"}
    assert compatibility.supports_single_node
    assert compatibility.supports_multi_node
    assert compatibility.supports_http_readiness
    assert compatibility.supports_queue_backpressure
    assert compatibility.supports_coordinated_failure


@pytest.mark.parametrize("runtime_version", ["invalid", "0.20.0", "0.21.0rc1", "0.24.0", "1!0.21.0"])
def test_unsupported_vllm_versions_fail_before_runtime(runtime_version: str) -> None:
    with pytest.raises(UnsupportedServingRuntimeError, match="vLLM version"):
        resolve_vllm_compatibility(runtime_version)
