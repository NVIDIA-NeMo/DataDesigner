# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from data_designer.slurm.client import environment as environment_module
from data_designer.slurm.client import runtime as runtime_module
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.records import ClientErrorCode
from data_designer.slurm.client.runtime import ClientRuntimeInspectionError, ClientRuntimeInspector
from data_designer.slurm.contracts import InstalledDistribution, compute_canonical_json_sha256
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.preflight import SystemAllocationPreflight
from data_designer.slurm.state import AttemptManifest


@pytest.fixture
def runtime_sites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    base, user = tmp_path / "site-packages", tmp_path / "user-site"
    base.mkdir()
    user.mkdir()
    for name in (
        "aiohttp",
        "data-designer",
        "data-designer-config",
        "data-designer-engine",
        "data-designer-slurm",
        "pip",
    ):
        version = "3.14.3" if name == "aiohttp" else "1.0.0"
        metadata = base / f"{name.replace('-', '_')}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    metadata = user / "irrelevant-1.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: irrelevant\nVersion: 1.0.0\n")
    monkeypatch.setattr(sys, "path", [str(base), str(user)])
    monkeypatch.setattr(environment_module.site, "getsitepackages", lambda: [str(base)])
    monkeypatch.setattr(environment_module.site, "getusersitepackages", lambda: str(user))
    monkeypatch.setattr(environment_module.site, "ENABLE_USER_SITE", True)
    return base, user


@pytest.mark.parametrize("pythonpath", (False, True))
def test_runtime_inspection_matches_batch_without_user_site(
    runtime_sites: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    single_node_plan: ResolvedSlurmRunPlan,
    attempt_manifest: AttemptManifest,
    pythonpath: bool,
) -> None:
    base, user = runtime_sites
    if pythonpath:
        monkeypatch.setenv("PYTHONPATH", str(user))
    else:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    submitted = ClientRuntimeInspector().inspect()
    assert "irrelevant" not in {item.name for item in submitted.distributions}
    monkeypatch.setattr(sys, "path", [str(base)])
    monkeypatch.setattr(environment_module.site, "ENABLE_USER_SITE", False)
    plan = single_node_plan.model_copy(
        update={"client": single_node_plan.client.model_copy(update={"runtime": submitted})}
    )
    attempt_directory = (
        Path(plan.authored_config.path).parent
        / "shards"
        / plan.shards[0].shard_id
        / "attempts"
        / attempt_manifest.attempt_id
    )
    context = AllocationContext(
        plan=plan, shard=plan.shards[0], attempt=attempt_manifest, attempt_directory=attempt_directory
    )
    SystemAllocationPreflight.verify_client_runtime(context)


@pytest.mark.parametrize("name", ("aiohttp", "data-designer", "pip"))
def test_runtime_inspector_rejects_required_user_only_distribution(
    runtime_sites: tuple[Path, Path],
    name: str,
) -> None:
    base, user = runtime_sites
    metadata = next(base.glob(f"{name.replace('-', '_')}-*.dist-info"))
    metadata.rename(user / metadata.name)
    with pytest.raises(ClientRuntimeInspectionError, match="user-site disabled") as failure:
        ClientRuntimeInspector().inspect()
    assert name in str(failure.value)
    assert "install" in str(failure.value)


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


def test_runtime_inspector_preserves_sanitized_inventory_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_inventory(_path: Path | None) -> tuple[InstalledDistribution, ...]:
        raise ClientWorkerError(
            ClientErrorCode.DEPENDENCY_CONFLICT,
            "mutable installed distribution 'example-plugin' is forbidden",
        )

    monkeypatch.setattr(runtime_module, "inspect_distributions", fail_inventory)

    with pytest.raises(ClientRuntimeInspectionError, match="example-plugin"):
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
