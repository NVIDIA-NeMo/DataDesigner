# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from data_designer.slurm.config import ContainerMount
from data_designer.slurm.contracts import ArtifactReference, compute_canonical_json_sha256
from data_designer.slurm.launcher.collection import render_collection_script
from data_designer.slurm.launcher.renderer import render_generation_retry_script
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import CollectionPlan, CollectionShard, RetryPlan, RetryShard
from data_designer.slurm.state.destinations import CollectionDestinationResolver


def test_collection_renderer_uses_authorized_mounts_and_no_gpu_directives(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    workspace_mount = ContainerMount(source="/workspace", target="/workspace", read_only=False)
    output_mount = ContainerMount(
        source="/workspace/primary/runs/run-001",
        target="/exports",
        read_only=False,
    )
    mounts = (workspace_mount, output_mount)
    profile = multi_node_plan.selected_profile.profile.model_copy(update={"container_mounts": list(mounts)})
    selection = multi_node_plan.selected_profile.model_copy(
        update={
            "profile": profile,
            "profile_sha256": compute_canonical_json_sha256(profile.model_dump(mode="json")),
        }
    )
    plan = ResolvedSlurmRunPlan.model_validate_json(
        json.dumps(
            multi_node_plan.model_copy(update={"container_mounts": mounts, "selected_profile": selection}).model_dump(
                mode="json"
            )
        )
    )
    destination = CollectionDestinationResolver().resolve(plan)
    collection = CollectionPlan(
        schema_version=1,
        collection_id="collection-0001",
        run_id=plan.run_id,
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        resolved_plan=ArtifactReference(
            path="/workspace/primary/runs/run-001/resolved-plan.json",
            sha256=plan.compute_sha256(),
        ),
        planned_shards=(
            CollectionShard(
                shard_id="shard-00000",
                winner_manifest=ArtifactReference(
                    path="/workspace/primary/runs/run-001/shards/shard-00000/winner.json",
                    sha256="a" * 64,
                ),
            ),
        ),
        host_destination=destination.host_path,
        container_destination=destination.container_path,
        num_partitions=plan.output.partitions,
    )

    script = render_collection_script(plan, collection, destination)

    assert 'readonly DD_STATE_MOUNT="/workspace/primary:/workspace/primary"' in script
    assert 'readonly DD_OUTPUT_MOUNT="/workspace/primary/runs/run-001:/exports"' in script
    assert (
        'readonly DD_COLLECTION_PLAN="/workspace/primary/runs/run-001/collections/collection-0001/plan.json"' in script
    )
    assert "data_designer.slurm.state.collection_worker" in script
    assert "#SBATCH --gres=" not in script
    assert "#SBATCH --gpus=" not in script
    assert subprocess.run(("bash", "-n"), input=script, text=True, check=False).returncode == 0


def test_retry_renderer_waits_for_persisted_attempt_before_starting_runtime(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    retry = RetryPlan(
        schema_version=1,
        retry_id="retry-0001",
        run_id=multi_node_plan.run_id,
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        resolved_plan=ArtifactReference(
            path="/workspace/primary/runs/run-001/resolved-plan.json",
            sha256=multi_node_plan.compute_sha256(),
        ),
        planned_shards=(
            RetryShard(
                shard_id="shard-00001",
                attempt_id="attempt-0002",
                attempt_ordinal=2,
                array_task_index=1,
            ),
        ),
        effective_resume_mode="never",
    )

    script = render_generation_retry_script(multi_node_plan, retry)

    assert "#SBATCH --array=1%2" in script
    assert 'DD_ATTEMPT_ORDINAL="0002"' in script
    assert 'readonly DD_ATTEMPT_MANIFEST="${DD_ATTEMPT_DIR}/attempt.json"' in script
    assert script.index("DD_ATTEMPT_MANIFEST") < script.index("DD_RUNTIME_DIR")
    assert "data_designer.slurm.state.attempt_identity" in script
    assert '--array-job-id "${DD_ARRAY_JOB_ID}" --array-task-id "${DD_ARRAY_TASK_ID}"' in script
    assert script.index("data_designer.slurm.state.attempt_identity") < script.index("DD_RUNTIME_DIR")
    assert subprocess.run(("bash", "-n"), input=script, text=True, check=False).returncode == 0
