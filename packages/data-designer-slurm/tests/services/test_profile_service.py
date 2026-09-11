# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from data_designer.slurm.config import (
    ImageBuildProfile,
    ProfileSelectionSource,
    SchedulerProfile,
    SlurmProfile,
    SlurmProfileCatalog,
    load_profile_catalog,
)
from data_designer.slurm.launcher.errors import SlurmCommandOutputError
from data_designer.slurm.services import SlurmServiceError, SlurmServiceErrorCode, create_slurm_profile_service


class _Launcher:
    def __init__(self, gpu_counts: tuple[int, ...] = ()) -> None:
        self.gpu_counts = gpu_counts
        self.partitions: list[str | None] = []

    def query_gpu_counts(self, *, partition: str | None = None) -> tuple[int, ...]:
        self.partitions.append(partition)
        return self.gpu_counts


class _MalformedLauncher(_Launcher):
    def query_gpu_counts(self, *, partition: str | None = None) -> tuple[int, ...]:
        del partition
        raise SlurmCommandOutputError("malformed")


def test_profile_init_creates_deterministic_private_starter_without_side_effects(tmp_path: Path) -> None:
    profile_file = tmp_path / "profile.yml"
    workspace = tmp_path / "workspace"
    launcher = _Launcher()
    service = create_slurm_profile_service(
        profile_file=profile_file,
        launcher=launcher,  # type: ignore[arg-type]
        hostname_resolver=lambda: ("Ignored.EXAMPLE.test",),
    )

    result = service.initialize(
        workspace_root=workspace,
        image_build_partition="cpu",
        cluster="primary",
        account="research",
        partition="gpu",
        host_patterns=("Login.EXAMPLE.test",),
    )

    assert result.profile_file == profile_file.as_posix()
    assert result.validation_command == f"data-designer slurm profile validate --profile-file {profile_file.as_posix()}"
    assert stat.S_IMODE(profile_file.stat().st_mode) == 0o600
    assert not workspace.exists()
    assert launcher.partitions == []
    assert load_profile_catalog(profile_file).model_dump(mode="json") == {
        "schema_version": 1,
        "default_cluster": "primary",
        "clusters": {
            "primary": {
                "schema_version": 1,
                "host_patterns": ["Login.EXAMPLE.test"],
                "scheduler": {
                    "account": "research",
                    "partition": "gpu",
                    "mem_per_gpu": None,
                    "bin_path": None,
                },
                "gpus_per_node": "auto",
                "workspace_root": workspace.as_posix(),
                "image_build": {
                    "partition": "cpu",
                    "cpus_per_task": 2,
                    "memory": "8G",
                    "time_limit": "04:00:00",
                },
                "gpu_request_mode": "gres",
                "container_mounts": [],
            }
        },
    }
    assert "container_mounts" not in profile_file.read_text()


def test_profile_init_uses_explicit_environment_then_home_path_precedence(tmp_path: Path) -> None:
    environment_file = tmp_path / "environment.yml"
    explicit_file = tmp_path / "explicit.json"
    environment = {"DATA_DESIGNER_SLURM_PROFILE_FILE": environment_file.as_posix()}

    explicit = create_slurm_profile_service(
        profile_file=explicit_file,
        environ=environment,
        home_directory=tmp_path,
    ).initialize(
        workspace_root=tmp_path / "workspace",
        image_build_partition="cpu",
        host_patterns=("login",),
    )
    from_environment = create_slurm_profile_service(
        environ=environment,
        home_directory=tmp_path,
    ).initialize(
        workspace_root=tmp_path / "workspace",
        image_build_partition="cpu",
        host_patterns=("login",),
    )
    from_home = create_slurm_profile_service(
        environ={},
        home_directory=tmp_path,
        hostname_resolver=lambda: ("Node.EXAMPLE.test", "node.example.test", "node-short"),
    ).initialize(
        workspace_root=tmp_path / "workspace",
        image_build_partition="cpu",
    )

    assert explicit.profile_file == explicit_file.as_posix()
    assert from_environment.profile_file == environment_file.as_posix()
    assert from_home.profile_file == (tmp_path / ".data-designer-slurm-profile.yml").as_posix()
    assert yaml.safe_load(explicit_file.read_text()) == json.loads(explicit_file.read_text())
    assert load_profile_catalog(from_home.profile_file).clusters["default"].host_patterns == [
        "node.example.test",
        "node-short",
    ]


def test_profile_init_refuses_to_overwrite_or_leave_temporary_files(tmp_path: Path) -> None:
    profile_file = tmp_path / "profile.yml"
    profile_file.write_text("owned by caller\n")

    with pytest.raises(SlurmServiceError) as caught:
        create_slurm_profile_service(profile_file=profile_file).initialize(
            workspace_root=tmp_path / "workspace",
            image_build_partition="cpu",
            host_patterns=("login",),
        )

    assert caught.value.code is SlurmServiceErrorCode.CONFLICT
    assert profile_file.read_text() == "owned by caller\n"
    assert tuple(tmp_path.iterdir()) == (profile_file,)


def test_profile_init_refuses_to_follow_dangling_destination_symlink(tmp_path: Path) -> None:
    profile_file = tmp_path / "profile.yml"
    target = tmp_path / "target.yml"
    profile_file.symlink_to(target)

    with pytest.raises(SlurmServiceError) as caught:
        create_slurm_profile_service(profile_file=profile_file).initialize(
            workspace_root=tmp_path / "workspace",
            image_build_partition="cpu",
            host_patterns=("login",),
        )

    assert caught.value.code is SlurmServiceErrorCode.CONFLICT
    assert profile_file.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize(
    "profile_file",
    (Path("profile.txt"), Path("missing/profile.yml")),
    ids=("unsupported-suffix", "missing-parent"),
)
def test_profile_init_rejects_invalid_destination(tmp_path: Path, profile_file: Path) -> None:
    with pytest.raises(SlurmServiceError) as caught:
        create_slurm_profile_service(profile_file=tmp_path / profile_file).initialize(
            workspace_root=tmp_path / "workspace",
            image_build_partition="cpu",
            host_patterns=("login",),
        )

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST


def test_profile_validate_uses_strict_loader_selection_and_effective_checks(tmp_path: Path) -> None:
    selected_workspace = tmp_path / "primary"
    selected_workspace.mkdir()
    catalog = _catalog(selected_workspace, tmp_path / "unused")
    profile_file = tmp_path / "profile.yml"
    profile_file.write_text(yaml.safe_dump(catalog.model_dump(mode="json"), sort_keys=False))
    launcher = _Launcher((8, 8))

    result = create_slurm_profile_service(
        profile_file=profile_file,
        launcher=launcher,  # type: ignore[arg-type]
        hostname_resolver=lambda: ("LOGIN-01.EXAMPLE.TEST", "login-01.example.test"),
    ).validate()

    assert result.profile_file == profile_file.as_posix()
    assert result.hostnames == ("login-01.example.test",)
    assert result.default_cluster == "fallback"
    assert result.selected_cluster == "primary"
    assert result.selection_source is ProfileSelectionSource.HOSTNAME
    assert result.matched_pattern == "login-*.example.test"
    assert [(match.cluster, match.patterns) for match in result.matched_clusters] == [
        ("primary", ("login-*.example.test",))
    ]
    assert result.workspace_root == selected_workspace.as_posix()
    assert result.image_root == (selected_workspace / "images").as_posix()
    assert result.registry_file == (selected_workspace / "images" / "registry.yaml").as_posix()
    assert result.gpus_per_node == 8
    assert launcher.partitions == ["gpu"]
    assert tuple(selected_workspace.iterdir()) == ()
    assert not (tmp_path / "unused").exists()


def test_profile_validate_honors_explicit_cluster_and_fixed_gpu_count(tmp_path: Path) -> None:
    workspace = tmp_path / "fallback"
    workspace.mkdir()
    catalog = _catalog(tmp_path / "unused", workspace)
    launcher = _Launcher()

    result = create_slurm_profile_service(
        catalog=catalog,
        cluster="fallback",
        launcher=launcher,  # type: ignore[arg-type]
        hostname_resolver=lambda: ("login-01.example.test",),
    ).validate()

    assert result.selected_cluster == "fallback"
    assert result.selection_source is ProfileSelectionSource.EXPLICIT
    assert result.gpus_per_node == 4
    assert launcher.partitions == []
    assert [(match.cluster, match.patterns) for match in result.matched_clusters] == [
        ("primary", ("login-*.example.test",))
    ]


@pytest.mark.parametrize("gpu_counts", ((), (4, 8)), ids=("missing", "heterogeneous"))
def test_profile_validate_rejects_ambiguous_automatic_gpu_count(
    tmp_path: Path,
    gpu_counts: tuple[int, ...],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(SlurmServiceError) as caught:
        create_slurm_profile_service(
            profile=_profile(workspace),
            launcher=_Launcher(gpu_counts),  # type: ignore[arg-type]
        ).validate()

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert str(caught.value) == "eligible Slurm nodes do not report one GPU count"


def test_profile_validate_rejects_ambiguous_selection_and_unavailable_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    overlapping = SlurmProfileCatalog(
        schema_version=1,
        default_cluster="first",
        clusters={
            "first": _profile(workspace, host_patterns=["node-*"]),
            "second": _profile(workspace, host_patterns=["*.example.test"]),
        },
    )

    with pytest.raises(SlurmServiceError) as ambiguous:
        create_slurm_profile_service(
            catalog=overlapping,
            hostname_resolver=lambda: ("node-01.example.test",),
        ).validate()

    missing = create_slurm_profile_service(
        profile=_profile(tmp_path / "missing", gpus_per_node=4),
        hostname_resolver=lambda: ("login",),
    )
    with pytest.raises(SlurmServiceError) as unavailable:
        missing.validate()

    assert ambiguous.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert unavailable.value.code is SlurmServiceErrorCode.UNAVAILABLE


def test_profile_validate_rejects_duplicate_keys_with_stable_error(tmp_path: Path) -> None:
    profile_file = tmp_path / "profile.yml"
    profile_file.write_text("schema_version: 1\nschema_version: 1\n")

    with pytest.raises(SlurmServiceError) as caught:
        create_slurm_profile_service(profile_file=profile_file).validate()

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert str(caught.value) == "profile configuration cannot be resolved"


def test_profile_validate_reports_malformed_slurm_output_as_unavailable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(SlurmServiceError) as caught:
        create_slurm_profile_service(
            profile=_profile(workspace),
            launcher=_MalformedLauncher(),  # type: ignore[arg-type]
        ).validate()

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert str(caught.value) == "Slurm is unavailable"


def _catalog(primary_workspace: Path, fallback_workspace: Path) -> SlurmProfileCatalog:
    return SlurmProfileCatalog(
        schema_version=1,
        default_cluster="fallback",
        clusters={
            "primary": _profile(primary_workspace, host_patterns=["login-*.example.test"]),
            "fallback": _profile(fallback_workspace, gpus_per_node=4),
        },
    )


def _profile(
    workspace: Path,
    *,
    host_patterns: list[str] | None = None,
    gpus_per_node: int | str = "auto",
) -> SlurmProfile:
    return SlurmProfile(
        schema_version=1,
        host_patterns=[] if host_patterns is None else host_patterns,
        scheduler=SchedulerProfile(partition="gpu"),
        gpus_per_node=gpus_per_node,
        workspace_root=workspace.as_posix(),
        image_build=ImageBuildProfile(
            partition="cpu",
            cpus_per_task=2,
            memory="8G",
            time_limit="04:00:00",
        ),
    )
