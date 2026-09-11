# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-authorized host and container collection destinations."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path

from data_designer.slurm.config import ContainerMount
from data_designer.slurm.contracts import is_path_below, paths_overlap, validate_absolute_path
from data_designer.slurm.images.records import validate_enroot_mount_path
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.errors import StateConflictError
from data_designer.slurm.state.outputs import CollectionPlan


@dataclass(frozen=True, slots=True)
class CollectionDestination:
    """One normalized output directory and its authorized writable mount."""

    host_path: str
    container_path: str
    mount: ContainerMount


class CollectionDestinationResolver:
    """Resolve the pinned output root through the selected writable mount map."""

    def resolve(
        self,
        plan: ResolvedSlurmRunPlan,
        requested_destination: str | Path | None = None,
    ) -> CollectionDestination:
        """Return one authorized output path in host and container namespaces."""
        raw_destination = plan.output.root if requested_destination is None else Path(requested_destination).as_posix()
        try:
            host_path = validate_absolute_path(raw_destination)
        except ValueError as error:
            raise StateConflictError("collection destination must be a normalized absolute path") from error
        workspace_root = plan.selected_profile.profile.workspace_root
        _validate_destination_overlap(plan, host_path)
        workspace_mount = ContainerMount(source=workspace_root, target=workspace_root, read_only=False)
        authorized = {
            (mount.source, mount.target): mount
            for mount in (workspace_mount, *plan.container_mounts)
            if not mount.read_only
        }
        writable = tuple(
            mount
            for mount in authorized.values()
            if host_path == mount.source or is_path_below(host_path, mount.source)
        )
        if not writable:
            raise StateConflictError("collection destination is not covered by a profile-authorized writable mount")
        longest = max(len(mount.source) for mount in writable)
        matches = tuple(mount for mount in writable if len(mount.source) == longest)
        if len(matches) != 1:
            raise StateConflictError("collection destination has an ambiguous writable mount mapping")
        mount = matches[0]
        if host_path == mount.source:
            raise StateConflictError("collection destination must be below its writable mount source")
        try:
            validate_enroot_mount_path(workspace_root)
            validate_enroot_mount_path(mount.source)
            validate_enroot_mount_path(mount.target)
        except ValueError as error:
            raise StateConflictError("collection paths cannot be represented as safe Enroot mounts") from error
        relative = posixpath.relpath(host_path, mount.source)
        container_path = mount.target if relative == "." else posixpath.join(mount.target, relative)
        return CollectionDestination(host_path, validate_absolute_path(container_path), mount)

    def validate_persisted(
        self,
        resolved_plan: ResolvedSlurmRunPlan,
        collection_plan: CollectionPlan,
    ) -> CollectionDestination:
        """Reauthorize persisted collection intent against its pinned run plan."""
        destination = self.resolve(resolved_plan, collection_plan.host_destination)
        if collection_plan.run_id != resolved_plan.run_id:
            raise StateConflictError("collection run identity does not match the resolved plan")
        if collection_plan.host_destination != destination.host_path:
            raise StateConflictError("collection host destination no longer matches resolved intent")
        if collection_plan.container_destination != destination.container_path:
            raise StateConflictError("collection container destination no longer matches resolved intent")
        if collection_plan.num_partitions != resolved_plan.output.partitions:
            raise StateConflictError("collection partition count no longer matches resolved intent")
        return destination


def _validate_destination_overlap(plan: ResolvedSlurmRunPlan, host_path: str) -> None:
    workspace_root = plan.selected_profile.profile.workspace_root
    reserved = tuple(posixpath.join(workspace_root, name) for name in ("images", "runtime", "benchmarks"))
    if any(paths_overlap(host_path, path) for path in reserved):
        raise StateConflictError("collection destination must not overlap package-managed workspace state")
    managed_assets_path = plan.invocation.effective_input_bindings.managed_assets_path
    assert managed_assets_path is not None
    if paths_overlap(host_path, managed_assets_path):
        raise StateConflictError("collection destination must not overlap managed assets")
    runs_root = posixpath.join(workspace_root, "runs")
    run_output_root = posixpath.join(runs_root, plan.run_id, "output")
    if paths_overlap(host_path, runs_root) and not (
        host_path == run_output_root or is_path_below(host_path, run_output_root)
    ):
        raise StateConflictError("collection destination must not overlap package-managed run state")


__all__ = ["CollectionDestination", "CollectionDestinationResolver"]
