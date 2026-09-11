# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic expansion of benchmark intent into ordinary run configs."""

from __future__ import annotations

import math
import posixpath

from pydantic import Field

from data_designer.slurm.config import (
    AdaptiveRecordPolicy,
    BenchmarkDeploymentCase,
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
)
from data_designer.slurm.contracts import ContractValue, Identifier, compute_canonical_json_sha256


class CompiledBenchmarkCase(ContractValue):
    """One stable benchmark case and its ordinary authored run config."""

    case_id: Identifier
    child_run_id: Identifier
    child_run_config: DataDesignerSlurmConfig


class CompiledBenchmark(ContractValue):
    """Ordered immutable result of benchmark expansion."""

    benchmark_id: Identifier
    cases: tuple[CompiledBenchmarkCase, ...] = Field(min_length=1)


class BenchmarkCompiler:
    """Compile a benchmark declaration without filesystem or scheduler access."""

    @staticmethod
    def compile(
        config: DataDesignerSlurmBenchmarkConfig,
        base_run: DataDesignerSlurmConfig,
    ) -> CompiledBenchmark:
        if not isinstance(config, DataDesignerSlurmBenchmarkConfig):
            raise TypeError("config must be a DataDesignerSlurmBenchmarkConfig")
        if not isinstance(base_run, DataDesignerSlurmConfig):
            raise TypeError("base_run must be a DataDesignerSlurmConfig")

        aliases = tuple(deployment.model_alias for deployment in base_run.deployments)
        selected_aliases = aliases if config.model_aliases == "all" else tuple(config.model_aliases)
        unknown_aliases = set(selected_aliases).difference(aliases)
        if unknown_aliases:
            raise ValueError(f"benchmark references unknown model aliases: {', '.join(sorted(unknown_aliases))}")
        for deployment_case in config.deployment_cases:
            unknown_deployments = set(deployment_case.deployments).difference(aliases)
            if unknown_deployments:
                raise ValueError(
                    f"benchmark deployment case references unknown aliases: {', '.join(sorted(unknown_deployments))}"
                )

        identity_digest = compute_canonical_json_sha256(
            {
                "benchmark": config.model_dump(mode="json"),
                "base_run": base_run.model_dump(mode="json"),
            }
        )
        benchmark_id = f"benchmark-{identity_digest[:32]}"
        expanded: list[tuple[str, DataDesignerSlurmConfig]] = []
        for concurrency in config.concurrency_values:
            for deployment_case in config.deployment_cases:
                case_id = derive_benchmark_case_id(deployment_case.name, concurrency)
                expanded.append(
                    (
                        case_id,
                        _compile_child(
                            base_run,
                            benchmark_id=benchmark_id,
                            case_id=case_id,
                            concurrency=concurrency,
                            selected_aliases=selected_aliases,
                            deployment_case=deployment_case,
                            requested_records=resolve_requested_records(config, concurrency),
                        ),
                    )
                )

        cases = tuple(
            CompiledBenchmarkCase(
                case_id=case_id,
                child_run_id=f"run-{identity_digest[:16]}-{index:04d}-{child.compute_sha256()[:12]}",
                child_run_config=child,
            )
            for index, (case_id, child) in enumerate(expanded)
        )
        return CompiledBenchmark(benchmark_id=benchmark_id, cases=cases)


def _compile_child(
    base_run: DataDesignerSlurmConfig,
    *,
    benchmark_id: str,
    case_id: str,
    concurrency: int,
    selected_aliases: tuple[str, ...],
    deployment_case: BenchmarkDeploymentCase,
    requested_records: int,
) -> DataDesignerSlurmConfig:
    model_concurrency = dict(base_run.invocation.model_concurrency)
    model_concurrency.update(dict.fromkeys(selected_aliases, concurrency))
    invocation = base_run.invocation.model_copy(
        update={
            "num_records": requested_records,
            "model_concurrency": model_concurrency,
        }
    )
    deployments = []
    for deployment in base_run.deployments:
        override = deployment_case.deployments.get(deployment.model_alias)
        if override is None:
            deployments.append(deployment)
            continue
        deployments.append(
            deployment.model_copy(
                update={
                    "resources": deployment.resources.model_copy(update={"nodes": override.nodes}),
                    "topology": deployment.topology.model_copy(
                        update={"nodes_per_replica": override.nodes_per_replica}
                    ),
                }
            )
        )

    output_root = base_run.output.root
    output = base_run.output
    if output_root is not None:
        output = output.model_copy(update={"root": posixpath.join(output_root, benchmark_id, case_id)})
    return DataDesignerSlurmConfig.model_validate(
        base_run.model_copy(
            update={
                "name": _bounded_identifier(f"{base_run.name}-{case_id}"),
                "invocation": invocation,
                "deployments": deployments,
                "output": output,
            }
        ).model_dump(mode="python")
    )


def resolve_requested_records(config: DataDesignerSlurmBenchmarkConfig, concurrency: int) -> int:
    """Resolve the deterministic record count for one concurrency value."""
    policy = config.record_policy
    if not isinstance(policy, AdaptiveRecordPolicy):
        return policy.records
    scaled = math.ceil(concurrency * policy.records_per_concurrency)
    return min(policy.max_records, max(policy.base_records, scaled))


def derive_benchmark_case_id(case_name: str, concurrency: int) -> str:
    """Return the stable bounded identity for one cross-product case."""
    return _bounded_identifier(f"{case_name}-c{concurrency}")


def _bounded_identifier(value: str) -> str:
    if len(value) <= 128:
        return value
    digest = compute_canonical_json_sha256(value)[:12]
    return f"{value[:115]}-{digest}"


__all__ = [
    "BenchmarkCompiler",
    "CompiledBenchmark",
    "CompiledBenchmarkCase",
    "derive_benchmark_case_id",
    "resolve_requested_records",
]
