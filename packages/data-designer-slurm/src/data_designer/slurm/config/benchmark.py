# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from data_designer.slurm.config.run import DataDesignerSlurmConfig
from data_designer.slurm.contracts import (
    AuthoredConfig,
    Duration,
    Identifier,
    ModelAlias,
    SchemaVersion,
    validate_local_config_path,
)


class BenchmarkBaseRun(AuthoredConfig):
    source: str | None = None
    inline: DataDesignerSlurmConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_source(cls, value: object) -> object:
        if isinstance(value, str):
            return {"source": value}
        return value

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str | None) -> str | None:
        return None if value is None else validate_local_config_path(value)

    @model_validator(mode="after")
    def validate_base_run(self) -> BenchmarkBaseRun:
        if (self.source is None) == (self.inline is None):
            raise ValueError("base_run requires exactly one of source or inline")
        return self


class BenchmarkDeploymentOverride(AuthoredConfig):
    nodes: PositiveInt
    nodes_per_replica: PositiveInt

    @model_validator(mode="after")
    def validate_topology(self) -> BenchmarkDeploymentOverride:
        if self.nodes % self.nodes_per_replica:
            raise ValueError("nodes_per_replica must divide benchmark deployment nodes")
        return self


class BenchmarkDeploymentCase(AuthoredConfig):
    name: Identifier
    deployments: dict[ModelAlias, BenchmarkDeploymentOverride] = Field(min_length=1)


class FixedRecordPolicy(AuthoredConfig):
    type: Literal["fixed"]
    records: PositiveInt


class AdaptiveRecordPolicy(AuthoredConfig):
    type: Literal["adaptive"]
    base_records: PositiveInt
    max_records: PositiveInt
    records_per_concurrency: PositiveFloat

    @model_validator(mode="after")
    def validate_bounds(self) -> AdaptiveRecordPolicy:
        if self.max_records < self.base_records:
            raise ValueError("adaptive max_records must not be less than base_records")
        return self


BenchmarkRecordPolicy = Annotated[FixedRecordPolicy | AdaptiveRecordPolicy, Field(discriminator="type")]


class BenchmarkAnalysisTargets(AuthoredConfig):
    target_total_records: PositiveInt
    target_runtime: Duration


class DataDesignerSlurmBenchmarkConfig(AuthoredConfig):
    """Authored benchmark intent expanded into ordinary Slurm run configs."""

    schema_version: SchemaVersion
    name: Identifier
    base_run: BenchmarkBaseRun
    model_aliases: Literal["all"] | list[ModelAlias]
    concurrency_values: list[PositiveInt] = Field(min_length=1)
    deployment_cases: list[BenchmarkDeploymentCase] = Field(min_length=1)
    record_policy: BenchmarkRecordPolicy
    analysis: BenchmarkAnalysisTargets

    @model_validator(mode="after")
    def validate_benchmark(self) -> DataDesignerSlurmBenchmarkConfig:
        if isinstance(self.model_aliases, list):
            if not self.model_aliases:
                raise ValueError("model_aliases must not be empty")
            if len(self.model_aliases) != len(set(self.model_aliases)):
                raise ValueError("benchmark model aliases must be unique")
        if len(self.concurrency_values) != len(set(self.concurrency_values)):
            raise ValueError("benchmark concurrency values must be unique")
        case_names = [case.name for case in self.deployment_cases]
        if len(case_names) != len(set(case_names)):
            raise ValueError("benchmark deployment case names must be unique")
        return self
