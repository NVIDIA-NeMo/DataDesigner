# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from data_designer.slurm.benchmark import BenchmarkCompiler
from data_designer.slurm.config import (
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    load_benchmark_config,
    load_run_config,
)
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.resolution import resolve_slurm_config


@pytest.mark.parametrize("source", ["shipped", "guide"])
def test_benchmark_examples_compile_every_child_plan(
    source: str,
    tmp_path: Path,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    examples = Path(__file__).parents[2] / "examples"
    benchmark_file = examples / "benchmark.yaml"
    if source == "guide":
        guide = Path(__file__).parents[4] / "fern/versions/latest/pages/slurm/benchmarks.mdx"
        blocks = re.findall(r"```yaml\n(.*?)```", guide.read_text(), flags=re.DOTALL)
        assert len(blocks) == 2
        (tmp_path / "benchmark-run.yaml").write_text(blocks[0])
        benchmark_file = tmp_path / "benchmark.yaml"
        benchmark_file.write_text(blocks[1])
    config = load_benchmark_config(benchmark_file)
    assert config.base_run.source is not None
    base_run = load_run_config(benchmark_file.parent / config.base_run.source)
    assert base_run.builder.source is not None
    builder_payload = yaml.safe_load((examples / base_run.builder.source).read_text())
    compiled = BenchmarkCompiler.compile(config, base_run)
    image = single_node_plan.deployments[0].image.model_copy(
        update={"authored_ref": base_run.deployments[0].server.image}
    )

    assert len(compiled.cases) == 4
    for case in compiled.cases:
        plan = SlurmRunCompiler.compile(
            resolve_slurm_config(
                case.child_run_config,
                selected_profile=single_node_plan.selected_profile,
                client_runtime=single_node_plan.client.runtime,
                deployment_images=(image,),
                dependency_lock=dependency_lock_single,
                runtime_bundle=single_node_plan.runtime_bundle,
                run_id=case.child_run_id,
                package_version=single_node_plan.package_version,
                resolved_gpus_per_node=8,
                builder_payload=builder_payload,
            )
        )
        topology = plan.deployments[0].topology
        assert topology.tensor_parallel == 8
        assert topology.pipeline_parallel == 1
        assert topology.gpus_per_replica == 8
        assert topology.replica_count == 2
        assert topology.nodes_per_replica == case.child_run_config.deployments[0].topology.nodes_per_replica


@pytest.mark.parametrize("pipeline_parallel", [None, 2], ids=["default-pp1", "explicit-pp2"])
def test_benchmark_node_placement_preserves_pipeline_size_and_replica_count(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    pipeline_parallel: int | None,
) -> None:
    payload = authored_run_single.model_dump(mode="json")
    topology = payload["deployments"][0]["topology"]
    topology["tensor_parallel"] = 8 if pipeline_parallel is None else 4
    topology.pop("pipeline_parallel", None)
    if pipeline_parallel is not None:
        topology["pipeline_parallel"] = pipeline_parallel
    base_run = DataDesignerSlurmConfig.model_validate(payload)
    compiled = BenchmarkCompiler.compile(benchmark_config, base_run)

    for case in compiled.cases:
        plan = SlurmRunCompiler.compile(
            resolve_slurm_config(
                case.child_run_config,
                selected_profile=single_node_plan.selected_profile,
                client_runtime=single_node_plan.client.runtime,
                deployment_images=(single_node_plan.deployments[0].image,),
                dependency_lock=dependency_lock_single,
                runtime_bundle=single_node_plan.runtime_bundle,
                run_id=case.child_run_id,
                package_version=single_node_plan.package_version,
                resolved_gpus_per_node=8,
            )
        )
        resolved = plan.deployments[0].topology
        assert resolved.pipeline_parallel == (1 if pipeline_parallel is None else pipeline_parallel)
        assert resolved.gpus_per_replica == 8
        assert resolved.replica_count == 2


def test_compiler_expands_stable_ordered_ordinary_runs(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    compiled = BenchmarkCompiler.compile(benchmark_config, authored_run)
    equivalent = DataDesignerSlurmBenchmarkConfig.model_validate(benchmark_config.model_dump(mode="json"))

    assert compiled == BenchmarkCompiler.compile(equivalent, authored_run)
    assert tuple(case.case_id for case in compiled.cases) == (
        "two-independent-replicas-c32",
        "one-two-node-replica-c32",
        "two-independent-replicas-c64",
        "one-two-node-replica-c64",
        "two-independent-replicas-c128",
        "one-two-node-replica-c128",
    )
    assert len({case.child_run_id for case in compiled.cases}) == 6
    assert all(case.child_run_config.invocation.num_records == 1000 for case in compiled.cases)
    assert all(case.child_run_config.array_tasks.count == 1 for case in compiled.cases)
    assert all(case.child_run_config.array_tasks.max_concurrent is None for case in compiled.cases)
    assert [deployment.resources.nodes for deployment in compiled.cases[0].child_run_config.deployments] == [2, 1]
    assert [deployment.topology.nodes_per_replica for deployment in compiled.cases[1].child_run_config.deployments] == [
        2,
        1,
    ]
    assert compiled.cases[0].child_run_config.invocation.model_concurrency == {"generator": 32, "judge": 32}


def test_compiled_child_maps_are_immutable(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    compiled = BenchmarkCompiler.compile(benchmark_config, authored_run)

    with pytest.raises(TypeError, match="frozen dictionary"):
        compiled.cases[0].child_run_config.invocation.model_concurrency["generator"] = 1


def test_compiler_rejects_unknown_alias_at_the_boundary(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    invalid = benchmark_config.model_copy(update={"model_aliases": ["missing"]})

    with pytest.raises(ValueError, match="unknown model aliases"):
        BenchmarkCompiler.compile(invalid, authored_run)


@pytest.mark.parametrize(
    ("concurrency", "expected_records"),
    [(1, 1000), (2000, 2000), (6000, 5000)],
)
def test_adaptive_record_policy_is_bounded(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
    concurrency: int,
    expected_records: int,
) -> None:
    config = benchmark_config.model_copy(
        update={
            "concurrency_values": [concurrency],
            "deployment_cases": [benchmark_config.deployment_cases[0]],
        }
    )

    compiled = BenchmarkCompiler.compile(config, authored_run)

    assert compiled.cases[0].child_run_config.invocation.num_records == expected_records


def test_default_record_policy_scales_high_concurrency(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    payload = benchmark_config.model_dump(mode="json")
    payload.pop("record_policy")
    payload["concurrency_values"] = [512, 1024, 8192]
    payload["deployment_cases"] = payload["deployment_cases"][:1]
    config = DataDesignerSlurmBenchmarkConfig.model_validate(payload)

    compiled = BenchmarkCompiler.compile(config, authored_run)

    assert [case.child_run_config.invocation.num_records for case in compiled.cases] == [1000, 1024, 5000]
    assert all(case.child_run_config.array_tasks.count == 1 for case in compiled.cases)

    payload["record_policy"] = {"type": "adaptive", "records_per_concurrency": 4.0}
    override = DataDesignerSlurmBenchmarkConfig.model_validate(payload)
    overridden = BenchmarkCompiler.compile(override, authored_run)

    assert [case.child_run_config.invocation.num_records for case in overridden.cases] == [2048, 4096, 5000]
