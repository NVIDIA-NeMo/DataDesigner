# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from data_designer.slurm.client.dependencies import (
    ClientDependencyResolutionError,
    ClientDependencyResolver,
)
from data_designer.slurm.config import ClientDependencies
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import LockedPackage, ResolvedDependencyLock, ResolvedSlurmRunPlan


def test_empty_requirements_resolve_without_invoking_pip(
    tmp_path: Path,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    def unexpected_runner(*_args: object) -> None:
        raise AssertionError("pip must not run")

    resolver = ClientDependencyResolver(command_runner=unexpected_runner)
    dependencies = ClientDependencies(requirements=[])

    with resolver.resolve(
        dependencies,
        single_node_plan.client.image,
        run_root=tmp_path / "workspace/runs/run-001",
        source_root=tmp_path,
    ) as resolved:
        assert resolved.lock.authored_requirements == ()
        assert resolved.lock.overlay_packages == ()
        assert resolved.lock.client_image_sha256 == single_node_plan.client.image.sha256
        assert resolved.wheel_sources == ()
        assert resolved.lock_source is None


def test_inline_requirements_resolve_pure_wheels_and_omit_image_packages(
    tmp_path: Path,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    token = "DUMMY_INDEX_TOKEN"
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def download(command: tuple[str, ...], _cwd: Path, environment: dict[str, str]) -> None:
        calls.append((command, environment))
        destination = Path(next(item.removeprefix("--dest=") for item in command if item.startswith("--dest=")))
        (destination / "example_plugin-1.2.0-py3-none-any.whl").write_bytes(b"plugin")
        (destination / "pip-26.1-py3-none-any.whl").write_bytes(b"image package")

    dependencies = ClientDependencies(
        requirements=["example-plugin==1.2.0"],
        index_credentials={
            "private-index": {"type": "secret", "environment": "PACKAGE_INDEX_TOKEN"},
        },
    )
    resolver = ClientDependencyResolver(
        command_runner=download,
        environ={
            "PACKAGE_INDEX_TOKEN": token,
            "PIP_CONFIG_FILE": "/ambient/pip.conf",
            "PIP_EXTRA_INDEX_URL": "https://ambient.example/simple",
            "PIP_INDEX_URL": "https://ambient.example/simple",
            "SAFE_VALUE": "preserved",
        },
        python_executable="/python",
    )
    run_root = tmp_path / "workspace/runs/run-001"

    with resolver.resolve(
        dependencies,
        single_node_plan.client.image,
        run_root=run_root,
        source_root=tmp_path,
    ) as resolved:
        assert calls[0][0][0:4] == ("/python", "-m", "pip", "download")
        assert token not in calls[0][0]
        assert "--index-url=https://pypi.org/simple" in calls[0][0]
        assert calls[0][1] == {"PIP_CONFIG_FILE": os.devnull, "SAFE_VALUE": "preserved"}
        assert tuple(package.name for package in resolved.lock.overlay_packages) == ("example-plugin",)
        package = resolved.lock.overlay_packages[0]
        assert package.artifact.path == (run_root / "dependencies/example_plugin-1.2.0-py3-none-any.whl").as_posix()
        assert package.artifact.sha256 == hashlib.sha256(b"plugin").hexdigest()
        assert len(resolved.wheel_sources) == 1
        assert resolved.wheel_sources[0].is_file()


def test_inline_resolution_rejects_platform_specific_wheels(
    tmp_path: Path,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    def download(command: tuple[str, ...], _cwd: Path, _environment: dict[str, str]) -> None:
        destination = Path(next(item.removeprefix("--dest=") for item in command if item.startswith("--dest=")))
        (destination / "example_plugin-1.2.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"plugin")

    resolver = ClientDependencyResolver(command_runner=download)

    with (
        pytest.raises(ClientDependencyResolutionError, match="platform-independent"),
        resolver.resolve(
            ClientDependencies(requirements=["example-plugin==1.2.0"]),
            single_node_plan.client.image,
            run_root=tmp_path / "workspace/runs/run-001",
            source_root=tmp_path,
        ),
    ):
        pass


def test_supplied_lock_is_verified_and_rebound_to_run_inputs(
    tmp_path: Path,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    wheel = tmp_path / "example_plugin-1.2.0-py3-none-any.whl"
    wheel.write_bytes(b"plugin")
    image = single_node_plan.client.image.inspection_facts
    supplied = ResolvedDependencyLock(
        schema_version=1,
        resolver_version="external-1",
        python_abi=image.python_abi,
        client_image_sha256=single_node_plan.client.image.sha256,
        authored_requirements=("example-plugin==1.2.0",),
        image_distributions=tuple(sorted(image.distributions, key=lambda item: item.name)),
        overlay_packages=(
            LockedPackage(
                name="example-plugin",
                version="1.2.0",
                artifact=ArtifactReference(
                    path=wheel.as_posix(),
                    sha256=hashlib.sha256(b"plugin").hexdigest(),
                ),
            ),
        ),
    )
    lock_file = tmp_path / "lock.json"
    lock_file.write_text(supplied.serialize_json(), encoding="utf-8")
    run_root = tmp_path / "workspace/runs/run-001"

    with ClientDependencyResolver().resolve(
        ClientDependencies(requirements=None, lock_file="lock.json"),
        single_node_plan.client.image,
        run_root=run_root,
        source_root=tmp_path,
    ) as resolved:
        assert resolved.lock.authored_source == "lock.json"
        assert resolved.lock.source == ArtifactReference(
            path=(run_root / "inputs/lock.json").as_posix(),
            sha256=hashlib.sha256(supplied.serialize_json().encode()).hexdigest(),
        )
        assert (
            resolved.lock.overlay_packages[0].artifact.path
            == (run_root / "dependencies/example_plugin-1.2.0-py3-none-any.whl").as_posix()
        )
        assert resolved.wheel_sources == (wheel,)
        assert resolved.lock_source == lock_file


def test_supplied_lock_rejects_platform_specific_wheels(
    tmp_path: Path,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    wheel = tmp_path / "example_plugin-1.2.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    wheel.write_bytes(b"plugin")
    image = single_node_plan.client.image.inspection_facts
    supplied = ResolvedDependencyLock(
        schema_version=1,
        resolver_version="external-1",
        python_abi=image.python_abi,
        client_image_sha256=single_node_plan.client.image.sha256,
        authored_requirements=("example-plugin==1.2.0",),
        image_distributions=tuple(sorted(image.distributions, key=lambda item: item.name)),
        overlay_packages=(
            LockedPackage(
                name="example-plugin",
                version="1.2.0",
                artifact=ArtifactReference(
                    path=wheel.as_posix(),
                    sha256=hashlib.sha256(b"plugin").hexdigest(),
                ),
            ),
        ),
    )
    lock_file = tmp_path / "lock.json"
    lock_file.write_text(supplied.serialize_json(), encoding="utf-8")

    with (
        pytest.raises(ClientDependencyResolutionError, match="artifact differs"),
        ClientDependencyResolver().resolve(
            ClientDependencies(requirements=None, lock_file="lock.json"),
            single_node_plan.client.image,
            run_root=tmp_path / "workspace/runs/run-001",
            source_root=tmp_path,
        ),
    ):
        pass


def test_resolution_redacts_runner_failure_and_missing_credentials(
    tmp_path: Path,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    def fail(*_args: object) -> None:
        raise RuntimeError("DUMMY_SECRET_VALUE")

    resolver = ClientDependencyResolver(command_runner=fail)
    dependencies = ClientDependencies(requirements=["example-plugin==1.2.0"])
    with pytest.raises(ClientDependencyResolutionError, match="resolution failed") as captured:
        with resolver.resolve(
            dependencies,
            single_node_plan.client.image,
            run_root=tmp_path / "workspace/runs/run-001",
            source_root=tmp_path,
        ):
            pass
    assert "DUMMY_SECRET_VALUE" not in str(captured.value)

    credentialed = ClientDependencies(
        requirements=["example-plugin==1.2.0"],
        index_credentials={"private": {"type": "secret", "environment": "MISSING_TOKEN"}},
    )
    with pytest.raises(ClientDependencyResolutionError, match="credential is unavailable"):
        with resolver.resolve(
            credentialed,
            single_node_plan.client.image,
            run_root=tmp_path / "workspace/runs/run-002",
            source_root=tmp_path,
        ):
            pass
