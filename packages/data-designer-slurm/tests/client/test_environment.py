# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import ClientWorkerCase

from data_designer.plugins.registry import PluginRegistry
from data_designer.slurm.client.environment import ClientEnvironmentBuilder, inspect_distributions
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.plugins import discover_plugins
from data_designer.slurm.client.records import ClientErrorCode, ClientInstallerOutcome
from data_designer.slurm.config.images import InstalledDistribution
from data_designer.slurm.planning import ResolvedSlurmRunPlan


def test_environment_prepares_empty_verified_overlay(client_worker_case: ClientWorkerCase) -> None:
    def inventory(path: Path | None) -> tuple[InstalledDistribution, ...]:
        return client_worker_case.lock.image_distributions if path is None else ()

    prepared = ClientEnvironmentBuilder(inventory=inventory).prepare(
        client_worker_case.plan_path,
        shard_id=client_worker_case.plan.shards[0].shard_id,
        attempt_id="attempt-0001",
        attempt_dir=client_worker_case.attempt_dir,
    )

    assert prepared.installer_outcome is ClientInstallerOutcome.NOT_REQUIRED
    assert prepared.installed_distributions == client_worker_case.lock.image_distributions


def test_inspect_distributions_omits_path_for_active_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def distributions(**kwargs: object) -> tuple[()]:
        calls.append(kwargs)
        return ()

    monkeypatch.setattr("data_designer.slurm.client.environment.importlib.metadata.distributions", distributions)

    assert inspect_distributions(None) == ()
    assert calls == [{}]


def test_inspect_distributions_rejects_unhashed_direct_url(monkeypatch: pytest.MonkeyPatch) -> None:
    distribution = Mock(
        metadata={"Name": "example"},
        version="1.0.0",
    )
    distribution.read_text.return_value = '{"url":"https://example.test/example.whl"}'
    monkeypatch.setattr(
        "data_designer.slurm.client.environment.importlib.metadata.distributions",
        lambda: (distribution,),
    )

    with pytest.raises(ClientWorkerError) as error:
        inspect_distributions(None)

    assert error.value.code is ClientErrorCode.DEPENDENCY_CONFLICT


def test_environment_rejects_client_image_inventory_conflict(client_worker_case: ClientWorkerCase) -> None:
    conflicting = (InstalledDistribution(name="unexpected", version="1.0"),)

    with pytest.raises(ClientWorkerError) as error:
        ClientEnvironmentBuilder(inventory=lambda _: conflicting).prepare(
            client_worker_case.plan_path,
            shard_id=client_worker_case.plan.shards[0].shard_id,
            attempt_id="attempt-0001",
            attempt_dir=client_worker_case.attempt_dir,
        )

    assert error.value.code is ClientErrorCode.DEPENDENCY_CONFLICT


@pytest.mark.parametrize(
    ("shard_id", "attempt_id", "attempt_dir"),
    (
        ("../escaped", "attempt-0001", "../escaped/attempts/attempt-0001"),
        ("shard-00000", "attempt-0000", "shard-00000/attempts/attempt-0000"),
    ),
)
def test_environment_rejects_invalid_identity_before_creating_attempt(
    client_worker_case: ClientWorkerCase,
    shard_id: str,
    attempt_id: str,
    attempt_dir: str,
) -> None:
    path = client_worker_case.plan_path.parent / "shards" / attempt_dir

    with pytest.raises(ClientWorkerError) as error:
        ClientEnvironmentBuilder().prepare(
            client_worker_case.plan_path,
            shard_id=shard_id,
            attempt_id=attempt_id,
            attempt_dir=path,
        )

    assert error.value.code is ClientErrorCode.INVALID_INPUT
    assert not path.exists()


def test_environment_rejects_missing_locked_wheel(client_worker_case: ClientWorkerCase) -> None:
    plan_payload = client_worker_case.plan.model_dump(mode="json")
    lock_payload = client_worker_case.lock.model_dump(mode="json")
    wheel_path = client_worker_case.plan_path.parent / "dependencies" / "missing_plugin-1.0.0-py3-none-any.whl"
    lock_payload["authored_requirements"] = ["missing-plugin==1.0.0"]
    lock_payload["overlay_packages"] = [
        {
            "name": "missing-plugin",
            "version": "1.0.0",
            "artifact": {"path": wheel_path.as_posix(), "sha256": "a" * 64},
        }
    ]
    lock = client_worker_case.lock.model_validate_json(json.dumps(lock_payload))
    Path(client_worker_case.plan.client.dependency_lock.path).write_text(lock.serialize_json())
    plan_payload["client"]["dependency_lock"]["sha256"] = lock.compute_sha256()
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(plan_payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())

    def inventory(path: Path | None) -> tuple[InstalledDistribution, ...]:
        return lock.image_distributions if path is None else ()

    with pytest.raises(ClientWorkerError) as error:
        ClientEnvironmentBuilder(inventory=inventory).prepare(
            client_worker_case.plan_path,
            shard_id=plan.shards[0].shard_id,
            attempt_id="attempt-0001",
            attempt_dir=client_worker_case.attempt_dir,
        )

    assert error.value.code is ClientErrorCode.DEPENDENCY_ARTIFACT_MISSING


def test_environment_installs_verified_wheel_overlay(client_worker_case: ClientWorkerCase) -> None:
    plan_payload = client_worker_case.plan.model_dump(mode="json")
    lock_payload = client_worker_case.lock.model_dump(mode="json")
    wheel_path = client_worker_case.plan_path.parent / "dependencies" / "example_plugin-1.0.0-py3-none-any.whl"
    wheel_path.parent.mkdir()
    wheel_path.write_bytes(b"immutable wheel artifact")
    lock_payload["authored_requirements"] = ["example-plugin==1.0.0"]
    lock_payload["overlay_packages"] = [
        {
            "name": "example-plugin",
            "version": "1.0.0",
            "artifact": {
                "path": wheel_path.as_posix(),
                "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
            },
        }
    ]
    lock = client_worker_case.lock.model_validate_json(json.dumps(lock_payload))
    Path(client_worker_case.plan.client.dependency_lock.path).write_text(lock.serialize_json())
    plan_payload["client"]["dependency_lock"]["sha256"] = lock.compute_sha256()
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(plan_payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())
    commands: list[tuple[str, ...]] = []

    def inventory(path: Path | None) -> tuple[InstalledDistribution, ...]:
        if path is None:
            return lock.image_distributions
        return (InstalledDistribution(name="example-plugin", version="1.0.0"),) if commands else ()

    prepared = ClientEnvironmentBuilder(inventory=inventory, command_runner=commands.append).prepare(
        client_worker_case.plan_path,
        shard_id=plan.shards[0].shard_id,
        attempt_id="attempt-0001",
        attempt_dir=client_worker_case.attempt_dir,
    )

    assert prepared.installer_outcome is ClientInstallerOutcome.INSTALLED
    assert "--no-index" in commands[0]
    assert "--no-deps" in commands[0]


def test_discover_plugins_loads_verified_entry_point(
    fake_plugin_overlay: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(fake_plugin_overlay.as_posix())
    monkeypatch.setattr("data_designer.plugins.registry.PLUGINS_DISABLED", False)
    importlib.invalidate_caches()
    PluginRegistry.reset()

    plugins = discover_plugins((InstalledDistribution(name="fake-data-designer-plugin", version="1.0.0"),))

    assert len(plugins) == 1
    assert plugins[0].plugin_name == "fake-slurm-column"
    PluginRegistry.reset()
