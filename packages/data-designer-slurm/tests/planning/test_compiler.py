# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from data_designer.slurm.config import (
    BuilderInput,
    DataDesignerSlurmConfig,
    InputBindings,
    SlurmProfileCatalog,
    select_profile,
)
from data_designer.slurm.planning import (
    ArtifactReference,
    ConfigurationResolutionError,
    EffectiveDataDesignerSlurmConfig,
    PlanCompilationError,
    ResolvedDependencyLock,
    ResolvedSlurmRunPlan,
    compile_slurm_run_plan,
    resolve_slurm_config,
)

GOLDEN_DIRECTORY = Path(__file__).parents[1] / "contracts" / "golden"


def _resolve_fixture(
    authored: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    expected: ResolvedSlurmRunPlan,
    **updates: object,
) -> EffectiveDataDesignerSlurmConfig:
    values = {
        "selected_profile": expected.selected_profile,
        "client_image": expected.client.image,
        "deployment_images": tuple(deployment.image for deployment in expected.deployments),
        "dependency_lock": dependency_lock,
        "runtime_bundle": expected.runtime_bundle,
        "run_id": expected.run_id,
        "package_version": expected.package_version,
        "resolved_gpus_per_node": expected.resolved_gpus_per_node,
    }
    values.update(updates)
    return resolve_slurm_config(authored, **values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("authored_fixture", "lock_fixture", "plan_fixture", "golden_name"),
    [
        ("authored_run_single", "dependency_lock_single", "single_node_plan", "single_node_plan.json"),
        ("authored_run", "dependency_lock", "multi_node_plan", "multi_node_plan.json"),
    ],
)
def test_compiler_reproduces_plan_goldens_byte_for_byte(
    request: pytest.FixtureRequest,
    authored_fixture: str,
    lock_fixture: str,
    plan_fixture: str,
    golden_name: str,
) -> None:
    authored = request.getfixturevalue(authored_fixture)
    dependency_lock = request.getfixturevalue(lock_fixture)
    expected = request.getfixturevalue(plan_fixture)
    effective = _resolve_fixture(authored, dependency_lock, expected)

    first = compile_slurm_run_plan(effective)
    second = compile_slurm_run_plan(effective)

    assert first.serialize_json() == (GOLDEN_DIRECTORY / golden_name).read_text()
    assert first.serialize_canonical_json() == second.serialize_canonical_json()
    assert first.compute_sha256() == second.compute_sha256() == expected.compute_sha256()


def test_compiler_resolves_explicit_hostname_and_default_profile_selection(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    profile_catalog: SlurmProfileCatalog,
) -> None:
    selections = (
        select_profile(profile_catalog, cluster="primary", hostnames=("lab-login-1",)),
        select_profile(profile_catalog, hostnames=("PRIMARY-LOGIN-1",)),
        select_profile(profile_catalog, hostnames=("unmatched",)),
    )

    plans = tuple(
        compile_slurm_run_plan(
            _resolve_fixture(
                authored_run_single,
                dependency_lock_single,
                single_node_plan,
                selected_profile=selected,
            )
        )
        for selected in selections
    )

    assert {plan.selected_profile.selection_source.value for plan in plans} == {
        "explicit",
        "hostname",
        "default",
    }
    assert {plan.output.root for plan in plans} == {"/workspace/primary/runs/run-single/output"}
    assert all(plan.deployments[0].node_indices == (0,) for plan in plans)


def test_auto_gpu_resolution_is_explicit_and_scheduler_free(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    profile_catalog: SlurmProfileCatalog,
) -> None:
    selected = select_profile(profile_catalog, cluster="lab")
    runtime_bundle = single_node_plan.runtime_bundle.model_copy(
        update={"path": "/workspace/lab/runtime/runtime.tar.gz"}
    )

    with pytest.raises(ConfigurationResolutionError, match="auto gpus_per_node"):
        _resolve_fixture(
            authored_run_single,
            dependency_lock_single,
            single_node_plan,
            selected_profile=selected,
            runtime_bundle=runtime_bundle,
            resolved_gpus_per_node=None,
        )

    plan = compile_slurm_run_plan(
        _resolve_fixture(
            authored_run_single,
            dependency_lock_single,
            single_node_plan,
            selected_profile=selected,
            runtime_bundle=runtime_bundle,
            resolved_gpus_per_node=8,
        )
    )

    assert plan.resolved_gpus_per_node == 8
    assert plan.output.root == "/workspace/lab/runs/run-single/output"
    assert plan.submission.account == "lab"


def test_compiler_preserves_explicit_compatibility_run_values(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["invocation"]["run_config"] = {
        "buffer_size": 2048,
        "disable_early_shutdown": False,
        "max_conversation_restarts": 3,
        "otel_metrics_port": 24000,
        "shutdown_error_rate": 0.25,
    }
    authored = DataDesignerSlurmConfig.model_validate(payload)

    effective = _resolve_fixture(authored, dependency_lock_single, single_node_plan)

    assert effective.invocation.effective_run_config["buffer_size"] == 2048
    assert effective.invocation.effective_run_config["disable_early_shutdown"] is False
    assert effective.invocation.effective_run_config["max_conversation_restarts"] == 3
    assert effective.invocation.effective_run_config["otel_metrics_port"] == 24000
    assert effective.invocation.effective_run_config["shutdown_error_rate"] == 0.25


def test_compiler_rejects_tensor_parallelism_that_does_not_divide_gpu_shape(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["deployments"][0]["topology"]["tensor_parallel"] = 3
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(PlanCompilationError, match="tensor_parallel"):
        compile_slurm_run_plan(_resolve_fixture(authored, dependency_lock_single, single_node_plan))


@pytest.mark.parametrize(
    "output_update",
    [
        {"root": "/outside/output"},
        {"root": "/workspace/primary/images/output"},
        {"root": "/workspace/primary/runtime/output"},
        {"partitions": 9},
    ],
)
def test_resolution_rejects_invalid_output_destinations(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    output_update: dict[str, object],
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["output"].update(output_update)
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(ConfigurationResolutionError, match="output"):
        _resolve_fixture(authored, dependency_lock_single, single_node_plan)


def test_compiler_rejects_model_alias_missing_from_builder(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"]["model_configs"][0]["alias"] = "other"
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(PlanCompilationError, match="deployment alias"):
        compile_slurm_run_plan(_resolve_fixture(authored, dependency_lock_single, single_node_plan))


def test_compiler_rejects_otel_collision_before_runtime(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["invocation"]["run_config"] = {"otel_metrics_port": 17000}
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(PlanCompilationError, match="OTEL"):
        compile_slurm_run_plan(_resolve_fixture(authored, dependency_lock_single, single_node_plan))


def test_sharded_seed_inputs_have_stable_ranges_and_partition_digests(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    invocation = authored_run.invocation.model_copy(
        update={"input_bindings": InputBindings(seed_path="/datasets/seed.parquet")}
    )
    authored = authored_run.model_copy(update={"invocation": invocation})

    first = compile_slurm_run_plan(_resolve_fixture(authored, dependency_lock, multi_node_plan))
    second = compile_slurm_run_plan(_resolve_fixture(authored, dependency_lock, multi_node_plan))

    assert [(shard.record_range.start_index, shard.record_range.end_index_exclusive) for shard in first.shards] == [
        (0, 50),
        (50, 100),
    ]
    assert all(shard.input_partition is not None for shard in first.shards)
    assert first.shards == second.shards


def test_sourced_builder_is_resolved_to_one_digest_bound_run_input(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    builder_payload = authored_run_single.model_dump(mode="json")["builder"]["inline"]
    assert isinstance(builder_payload, dict)
    authored = authored_run_single.model_copy(update={"builder": BuilderInput(source="builder.json")})

    plan = compile_slurm_run_plan(
        _resolve_fixture(
            authored,
            dependency_lock_single,
            single_node_plan,
            builder_payload=builder_payload,
        )
    )

    assert plan.builder.authored_source == "builder.json"
    assert plan.builder.source is not None
    assert plan.builder.source.path == "/workspace/primary/runs/run-single/builder-config.json"
    assert plan.builder.content_sha256 == plan.builder.source.sha256


def test_resolution_rejects_artifact_identity_mismatches(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    wrong_lock = dependency_lock_single.model_copy(update={"client_image_sha256": "a" * 64})
    wrong_runtime = ArtifactReference(path="/tmp/runtime.tar.gz", sha256="e" * 64)

    with pytest.raises(ConfigurationResolutionError, match="client image"):
        _resolve_fixture(authored_run_single, wrong_lock, single_node_plan)
    with pytest.raises(ConfigurationResolutionError, match="runtime bundle"):
        _resolve_fixture(
            authored_run_single,
            dependency_lock_single,
            single_node_plan,
            runtime_bundle=wrong_runtime,
        )


def test_plan_contains_secret_references_without_credentials(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    plan = compile_slurm_run_plan(_resolve_fixture(authored_run, dependency_lock, multi_node_plan))
    serialized = plan.serialize_json()

    assert "PACKAGE_INDEX_TOKEN" in serialized
    assert "HF_TOKEN" in serialized
    assert "secret-value" not in serialized
