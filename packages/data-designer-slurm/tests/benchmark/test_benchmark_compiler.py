# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.slurm.benchmark import BenchmarkCompiler
from data_designer.slurm.config import DataDesignerSlurmBenchmarkConfig, DataDesignerSlurmConfig


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
