# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from data_designer.slurm.client import runtime as runtime_module
from data_designer.slurm.client.runtime import ClientRuntimeInspectionError, ClientRuntimeInspector
from data_designer.slurm.contracts import InstalledDistribution, compute_canonical_json_sha256


def test_runtime_inspector_binds_active_interpreter_and_required_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributions = tuple(
        InstalledDistribution(name=name, version="1.0.0")
        for name in (
            "aiohttp",
            "data-designer",
            "data-designer-config",
            "data-designer-engine",
            "data-designer-slurm",
            "pip",
        )
    )
    distributions = (
        InstalledDistribution(name="aiohttp", version="3.14.3"),
        *(distribution for distribution in distributions if distribution.name != "aiohttp"),
    )
    monkeypatch.setattr(runtime_module, "inspect_distributions", lambda path: distributions)

    runtime = ClientRuntimeInspector().inspect()

    assert runtime.python_executable == sys.executable
    assert runtime.distributions == distributions
    assert runtime.runtime_sha256 == compute_canonical_json_sha256(
        runtime.model_dump(mode="json", exclude={"runtime_sha256"})
    )


def test_runtime_inspector_rejects_missing_required_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "inspect_distributions",
        lambda path: (InstalledDistribution(name="pip", version="1.0.0"),),
    )

    with pytest.raises(ClientRuntimeInspectionError, match="required client distributions"):
        ClientRuntimeInspector().inspect()


def test_runtime_inspector_rejects_incompatible_proxy_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    distributions = tuple(
        InstalledDistribution(name=name, version="3.13.0" if name == "aiohttp" else "1.0.0")
        for name in (
            "aiohttp",
            "data-designer",
            "data-designer-config",
            "data-designer-engine",
            "data-designer-slurm",
            "pip",
        )
    )
    monkeypatch.setattr(runtime_module, "inspect_distributions", lambda path: distributions)

    with pytest.raises(ClientRuntimeInspectionError, match="aiohttp>=3.14.3,<4"):
        ClientRuntimeInspector().inspect()


def test_runtime_inspector_rejects_mixed_data_designer_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    distributions = tuple(
        InstalledDistribution(
            name=name,
            version=("3.14.3" if name == "aiohttp" else "0.9.3" if name == "data-designer-slurm" else "0.9.2"),
        )
        for name in (
            "aiohttp",
            "data-designer",
            "data-designer-config",
            "data-designer-engine",
            "data-designer-slurm",
            "pip",
        )
    )
    monkeypatch.setattr(runtime_module, "inspect_distributions", lambda path: distributions)

    with pytest.raises(ClientRuntimeInspectionError, match="must share one version"):
        ClientRuntimeInspector().inspect()
