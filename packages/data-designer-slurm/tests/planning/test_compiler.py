# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from data_designer.config import (
    DataDesignerConfigBuilder,
    DropColumnsProcessorConfig,
    JudgeScoreProfilerConfig,
    LLMTextColumnConfig,
    ModelConfig,
)
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


@pytest.mark.parametrize(
    ("run_config", "expected_rate", "expected_window"),
    [
        ({"shutdown_error_rate": 0.25}, 0.25, 10),
        ({"shutdown_error_window": 25}, 0.5, 25),
    ],
)
def test_compiler_preserves_partial_early_shutdown_configuration(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    run_config: dict[str, object],
    expected_rate: float,
    expected_window: int,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["invocation"]["run_config"] = run_config
    authored = DataDesignerSlurmConfig.model_validate(payload)

    effective = _resolve_fixture(authored, dependency_lock_single, single_node_plan)

    assert effective.invocation.effective_run_config["disable_early_shutdown"] is False
    assert effective.invocation.effective_run_config["shutdown_error_rate"] == expected_rate
    assert effective.invocation.effective_run_config["shutdown_error_window"] == expected_window


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
        {"root": "/workspace/primary/runs/other-run/output"},
        {"root": "/workspace/primary/runs/run-single/shards"},
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


def test_effective_config_rejects_invalid_direct_construction(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    effective = _resolve_fixture(authored_run, dependency_lock, multi_node_plan)
    values = {name: getattr(effective, name) for name in EffectiveDataDesignerSlurmConfig.model_fields}
    values["authored"] = authored_run.model_copy(
        update={"output": authored_run.output.model_copy(update={"format": "jsonl"})}
    )
    values["output"] = effective.output.model_copy(update={"format": "jsonl"})

    with pytest.raises(ValueError, match="parquet output"):
        EffectiveDataDesignerSlurmConfig(**values)  # type: ignore[arg-type]

    other_output = "/workspace/primary/runs/other-run/output"
    values["authored"] = authored_run.model_copy(
        update={"output": authored_run.output.model_copy(update={"root": other_output})}
    )
    values["output"] = effective.output.model_copy(update={"root": other_output})

    with pytest.raises(ValueError, match="another package-managed run"):
        EffectiveDataDesignerSlurmConfig(**values)  # type: ignore[arg-type]

    values["authored"] = authored_run.model_copy(
        update={"output": authored_run.output.model_copy(update={"partitions": 101})}
    )
    values["output"] = effective.output.model_copy(update={"partitions": 101})

    with pytest.raises(ValueError, match="requested records"):
        EffectiveDataDesignerSlurmConfig(**values)  # type: ignore[arg-type]

    values["authored"] = authored_run
    values["output"] = effective.output.model_copy(update={"format": "jsonl"})

    with pytest.raises(ValueError, match="resolved output"):
        EffectiveDataDesignerSlurmConfig(**values)  # type: ignore[arg-type]


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


@pytest.mark.parametrize("sourced", [False, True], ids=["inline", "sourced"])
def test_compiler_rejects_builder_model_alias_without_deployment(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    sourced: bool,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"]["model_configs"].append(
        {"alias": "undeployed", "model": "example/undeployed", "provider": "openai"}
    )
    builder_payload = None
    if sourced:
        builder_payload = payload["builder"]["inline"]
        payload["builder"] = {"source": "builder.json"}
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(PlanCompilationError, match="exactly cover"):
        compile_slurm_run_plan(
            _resolve_fixture(
                authored,
                dependency_lock_single,
                single_node_plan,
                builder_payload=builder_payload,
            )
        )


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


@pytest.mark.parametrize(
    ("builder_update", "output_update", "message"),
    [
        (
            {
                "seed_config": {
                    "sampling_strategy": "shuffle",
                    "source": {"path": "/datasets/seed.parquet", "seed_type": "local"},
                }
            },
            {},
            "shuffled seed",
        ),
        (
            {
                "seed_config": {
                    "sampling_strategy": "ordered",
                    "source": {"path": "/datasets/seed.parquet", "seed_type": "local"},
                }
            },
            {},
            "seed_path",
        ),
        (
            {
                "seed_config": {
                    "sampling_strategy": "ordered",
                    "selection_strategy": {"start": 0, "end": 9},
                    "source": {"path": "/datasets/seed.parquet", "seed_type": "local"},
                }
            },
            {},
            "selection strategies",
        ),
        (
            {
                "columns": [
                    {
                        "column_type": "image",
                        "model_alias": "generator",
                        "name": "picture",
                        "prompt": "an image",
                    }
                ]
            },
            {},
            "media output",
        ),
        ({}, {"format": "jsonl"}, "parquet output"),
    ],
)
def test_resolution_rejects_unshardable_big_iron_fields(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
    builder_update: dict[str, object],
    output_update: dict[str, object],
    message: str,
) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"].update(builder_update)
    payload["output"].update(output_update)
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(ConfigurationResolutionError, match=message):
        _resolve_fixture(authored, dependency_lock, multi_node_plan)


@pytest.mark.parametrize("field", ["processors", "profilers"])
def test_resolution_rejects_real_global_builder_configs(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
    field: str,
) -> None:
    builder = DataDesignerConfigBuilder(
        model_configs=[ModelConfig(alias="generator", model="example/generator", provider="openai")]
    )
    builder.add_column(LLMTextColumnConfig(name="generated", prompt="hello", model_alias="generator"))
    if field == "processors":
        builder.add_processor(DropColumnsProcessorConfig(name="global", column_names=["generated"]))
    else:
        builder.add_profiler(JudgeScoreProfilerConfig(model_alias="generator"))
    data_designer = builder.get_builder_config().to_dict()["data_designer"]
    payload = authored_run.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"][field] = data_designer[field]
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(ConfigurationResolutionError, match=field):
        _resolve_fixture(authored, dependency_lock, multi_node_plan)


@pytest.mark.parametrize("sourced", [False, True], ids=["inline", "sourced"])
@pytest.mark.parametrize(
    ("column", "message"),
    [
        ({"column_type": "future-column", "name": "future"}, "unknown column semantics"),
        ({"column_type": "fake-slurm-column", "name": "plugin"}, "plugin"),
        ({"column_type": "custom", "generator_function": "generate", "name": "custom"}, "custom"),
        (
            {
                "column_type": "validation",
                "name": "validate",
                "target_columns": ["generated"],
                "validator_params": {
                    "validation_function": "validate",
                    "validator_type": "local_callable",
                },
                "validator_type": "local_callable",
            },
            "local callable",
        ),
    ],
)
def test_resolution_rejects_unportable_multi_shard_columns(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
    sourced: bool,
    column: dict[str, object],
    message: str,
) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"]["columns"] = [column]
    builder_payload = None
    if sourced:
        builder_payload = payload["builder"]["inline"]
        payload["builder"] = {"source": "builder.json"}
    authored = DataDesignerSlurmConfig.model_validate(payload)

    with pytest.raises(ConfigurationResolutionError, match=message):
        _resolve_fixture(
            authored,
            dependency_lock,
            multi_node_plan,
            builder_payload=builder_payload,
        )


def test_sharded_seed_binding_may_override_authored_source(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"]["seed_config"] = {
        "sampling_strategy": "ordered",
        "source": {"path": "/datasets/original.parquet", "seed_type": "local"},
    }
    payload["invocation"]["input_bindings"]["seed_path"] = "/datasets/override.parquet"
    authored = DataDesignerSlurmConfig.model_validate(payload)

    plan = compile_slurm_run_plan(_resolve_fixture(authored, dependency_lock, multi_node_plan))

    assert all(shard.input_partition is not None for shard in plan.shards)


def test_single_shard_allows_non_collectable_big_iron_fields(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"].update(
        {
            "columns": [
                {
                    "column_type": "image",
                    "model_alias": "generator",
                    "name": "picture",
                    "prompt": "an image",
                }
            ],
            "seed_config": {
                "sampling_strategy": "shuffle",
                "source": {"path": "/datasets/seed.parquet", "seed_type": "local"},
            },
        }
    )
    payload["output"]["format"] = "jsonl"
    authored = DataDesignerSlurmConfig.model_validate(payload)

    plan = compile_slurm_run_plan(_resolve_fixture(authored, dependency_lock_single, single_node_plan))

    assert plan.array_tasks.count == 1
    assert plan.output.format == "jsonl"


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bytes", "builder digest"),
        ("aliases", "model aliases"),
    ],
)
def test_effective_config_rejects_sourced_builder_payload_identity_drift(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    mutation: str,
    message: str,
) -> None:
    builder_payload = authored_run_single.model_dump(mode="json")["builder"]["inline"]
    authored = authored_run_single.model_copy(update={"builder": BuilderInput(source="builder.json")})
    effective = _resolve_fixture(
        authored,
        dependency_lock_single,
        single_node_plan,
        builder_payload=builder_payload,
    )
    drifted_payload = deepcopy(builder_payload)
    if mutation == "bytes":
        drifted_payload["library_version"] = "drifted"
    else:
        drifted_payload["data_designer"]["model_configs"][0]["alias"] = "drifted"
    values = {name: getattr(effective, name) for name in EffectiveDataDesignerSlurmConfig.model_fields}
    values["builder_payload"] = drifted_payload

    with pytest.raises(ValueError, match=message):
        EffectiveDataDesignerSlurmConfig(**values)  # type: ignore[arg-type]


def test_resolution_rejects_secret_values_in_sourced_builder_payload(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    builder_payload = authored_run_single.model_dump(mode="json")["builder"]["inline"]
    secret = "super-secret-token"
    builder_payload["data_designer"]["api_key"] = secret
    authored = authored_run_single.model_copy(update={"builder": BuilderInput(source="builder.json")})

    with pytest.raises(ConfigurationResolutionError, match="secret values") as error:
        _resolve_fixture(
            authored,
            dependency_lock_single,
            single_node_plan,
            builder_payload=builder_payload,
        )

    assert secret not in str(error.value)
    assert error.value.__cause__ is not None
    assert secret not in str(error.value.__cause__)


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
