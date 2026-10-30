# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from slurm_test_fakes import (
    FakeClock,
    FakeCommandResponse,
    FakeLogicalEndpoint,
    FakeSlurmArray,
    FakeSlurmRunner,
    FakeSlurmTask,
    FakeVllmBackend,
)

from data_designer.config import SamplerColumnConfig, UUIDSamplerParams
from data_designer.config.models import ModelProvider
from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.client import ClientResult
from data_designer.slurm.config import (
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    DataDesignerSlurmConfigBuilder,
    ImageInspectionRecord,
    ServerDeploymentConfig,
    SlurmProfileCatalog,
)
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.runtime import network as runtime_network
from data_designer.slurm.state import (
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputManifest,
    CollectionPlan,
    RunManifest,
    SchedulerIdentity,
    SchedulerObservation,
    ShardManifest,
    ShardWinner,
)

TEST_DIRECTORY = Path(__file__).parent
CONTRACT_GOLDEN_DIRECTORY = TEST_DIRECTORY / "contracts" / "golden"
INTEGRATION_GOLDEN_PATH = TEST_DIRECTORY / "integration" / "golden" / "finalization_chain.json"
SLURM_GOLDEN_DIRECTORY = TEST_DIRECTORY / "slurm_test_fakes" / "golden" / "slurm"
STATE_GOLDEN_DIRECTORY = TEST_DIRECTORY / "state" / "golden"


@pytest.fixture(autouse=True)
def port_sysctls(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path_factory.mktemp("sysctl")
    (root / "ip_local_port_range").write_text("32768 60999\n")
    (root / "ip_local_reserved_ports").write_text("\n")
    (root / "ip_unprivileged_port_start").write_text("1024\n")
    monkeypatch.setattr(runtime_network, "_SYSCTL_ROOT", root)
    return root


@pytest.fixture
def authored_run() -> DataDesignerSlurmConfig:
    return DataDesignerSlurmConfig.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "authored_run.json").read_text())


@pytest.fixture
def authored_run_single() -> DataDesignerSlurmConfig:
    return DataDesignerSlurmConfig.model_validate_json(
        (CONTRACT_GOLDEN_DIRECTORY / "authored_run_single.json").read_text()
    )


@pytest.fixture
def profile_catalog() -> SlurmProfileCatalog:
    return SlurmProfileCatalog.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "profile_catalog.json").read_text())


@pytest.fixture
def dependency_lock() -> ResolvedDependencyLock:
    return ResolvedDependencyLock.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "dependency_lock.json").read_text())


@pytest.fixture
def dependency_lock_single() -> ResolvedDependencyLock:
    return ResolvedDependencyLock.model_validate_json(
        (CONTRACT_GOLDEN_DIRECTORY / "dependency_lock_single.json").read_text()
    )


@pytest.fixture
def single_node_plan() -> ResolvedSlurmRunPlan:
    return ResolvedSlurmRunPlan.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "single_node_plan.json").read_text())


@pytest.fixture(params=("empty", "omitted", "none"))
def model_free_run(
    request: pytest.FixtureRequest, authored_run_single: DataDesignerSlurmConfig
) -> DataDesignerSlurmConfig:
    payload = authored_run_single.model_dump(mode="json")
    builder = payload["builder"]["inline"]["data_designer"]
    if request.param == "omitted":
        builder.pop("model_configs")
    else:
        builder["model_configs"] = [] if request.param == "empty" else None
    builder["columns"] = [
        SamplerColumnConfig(name="id", sampler_type="uuid", params=UUIDSamplerParams()).model_dump(mode="json")
    ]
    payload["deployments"] = []
    payload["invocation"]["model_concurrency"] = {}
    return DataDesignerSlurmConfig.model_validate_json(json.dumps(payload))


@pytest.fixture(params=("mixed", "hosted-only"))
def external_run(
    request: pytest.FixtureRequest, authored_run_single: DataDesignerSlurmConfig
) -> DataDesignerSlurmConfig:
    payload = authored_run_single.model_dump(mode="python")
    models = payload["builder"]["inline"]["data_designer"]["model_configs"]
    models.append(
        {
            "alias": "judge",
            "model": "example/hosted-judge",
            "provider": "external",
            "inference_parameters": {
                "temperature": 0.3,
                "max_parallel_requests": 7,
                "extra_body": {"reasoning_effort": "medium"},
            },
        }
    )
    payload["invocation"]["model_concurrency"]["judge"] = 3
    if request.param == "hosted-only":
        models.pop(0)
        payload["deployments"] = []
        del payload["invocation"]["model_concurrency"]["generator"]
    return DataDesignerSlurmConfig.model_validate(payload)


@pytest.fixture
def hosted_model_providers() -> tuple[ModelProvider, ...]:
    return (
        ModelProvider(
            name="external",
            endpoint="https://models.example/v1",
            api_key="MODEL_API_KEY",
            extra_body={"routing": {"version": 2}},
            extra_headers={"X-Project": "test"},
        ),
    )


@pytest.fixture
def external_plan(
    hosted_model_providers: tuple[ModelProvider, ...],
    external_run: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
) -> ResolvedSlurmRunPlan:
    return SlurmRunCompiler.compile(
        resolve_slurm_config(
            external_run,
            hosted_model_providers=hosted_model_providers,
            selected_profile=single_node_plan.selected_profile,
            client_runtime=single_node_plan.client.runtime,
            deployment_images=tuple(single_node_plan.deployments[0].image for _ in external_run.deployments),
            dependency_lock=dependency_lock_single,
            runtime_bundle=single_node_plan.runtime_bundle,
            run_id=single_node_plan.run_id,
            package_version=single_node_plan.package_version,
        )
    )


@pytest.fixture
def shared_node_config(authored_run_single: DataDesignerSlurmConfig) -> DataDesignerSlurmConfig:
    payload = authored_run_single.model_dump(mode="python")
    models = (("writer", "Qwen/Qwen2.5-0.5B-Instruct"), ("judge", "microsoft/Phi-3-mini-4k-instruct"))
    payload["builder"]["inline"]["data_designer"]["model_configs"] = [
        {"alias": alias, "model": model, "provider": "openai"} for alias, model in models
    ]
    builder = DataDesignerSlurmConfigBuilder.from_builder_source("builder.yaml").with_invocation(
        num_records=4, dataset_name="judged", model_concurrency={"writer": 1, "judge": 1}
    )
    for alias, model in models:
        deployment = authored_run_single.deployments[0].model_dump(mode="python")
        deployment.update(
            model_alias=alias, model=model, resources={"nodes": 1, "shared_node": "models", "gpus_per_node": 4}
        )
        deployment["topology"]["tensor_parallel"] = 2
        builder.with_deployment(ServerDeploymentConfig.model_validate(deployment))
    config = builder.build().model_dump(mode="python")
    config["builder"] = payload["builder"]
    return DataDesignerSlurmConfig.model_validate(config)


@pytest.fixture
def shared_node_plan(
    shared_node_config: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
) -> ResolvedSlurmRunPlan:
    return SlurmRunCompiler.compile(
        resolve_slurm_config(
            shared_node_config,
            selected_profile=single_node_plan.selected_profile,
            client_runtime=single_node_plan.client.runtime,
            deployment_images=(single_node_plan.deployments[0].image,) * 2,
            dependency_lock=dependency_lock_single,
            runtime_bundle=single_node_plan.runtime_bundle,
            run_id=single_node_plan.run_id,
            package_version=single_node_plan.package_version,
        )
    )


@pytest.fixture
def multi_node_plan() -> ResolvedSlurmRunPlan:
    return ResolvedSlurmRunPlan.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "multi_node_plan.json").read_text())


@pytest.fixture
def client_result() -> ClientResult:
    return ClientResult.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "client_result.json").read_text())


@pytest.fixture
def benchmark_config() -> DataDesignerSlurmBenchmarkConfig:
    return DataDesignerSlurmBenchmarkConfig.model_validate_json(
        (CONTRACT_GOLDEN_DIRECTORY / "benchmark_config.json").read_text()
    )


@pytest.fixture
def benchmark_manifest() -> BenchmarkManifest:
    return BenchmarkManifest.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "benchmark_manifest.json").read_text())


@pytest.fixture
def benchmark_report() -> BenchmarkReport:
    return BenchmarkReport.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "benchmark_report.json").read_text())


@pytest.fixture
def client_image_inspection() -> ImageInspectionRecord:
    return ImageInspectionRecord.model_validate_json(
        (CONTRACT_GOLDEN_DIRECTORY / "client_image_inspection.json").read_text()
    )


@pytest.fixture
def serving_image_inspection() -> ImageInspectionRecord:
    return ImageInspectionRecord.model_validate_json(
        (CONTRACT_GOLDEN_DIRECTORY / "serving_image_inspection.json").read_text()
    )


@pytest.fixture
def collection_plan() -> CollectionPlan:
    return CollectionPlan.model_validate_json((STATE_GOLDEN_DIRECTORY / "collection_plan.json").read_text())


@pytest.fixture
def accounting_lag_observation() -> SchedulerObservation:
    return SchedulerObservation.model_validate_json((STATE_GOLDEN_DIRECTORY / "accounting_lag.json").read_text())


@pytest.fixture
def finalization_chain_payload() -> dict[str, object]:
    return cast(dict[str, object], json.loads(INTEGRATION_GOLDEN_PATH.read_text()))


@pytest.fixture
def run_manifest(finalization_chain_payload: dict[str, object]) -> RunManifest:
    return RunManifest.model_validate_json(json.dumps(finalization_chain_payload["run"]))


@pytest.fixture
def shard_manifests(finalization_chain_payload: dict[str, object]) -> tuple[ShardManifest, ...]:
    return tuple(
        ShardManifest.model_validate_json(json.dumps(payload))
        for payload in cast(list[object], finalization_chain_payload["shards"])
    )


@pytest.fixture
def attempt_manifest(finalization_chain_payload: dict[str, object]) -> AttemptManifest:
    return AttemptManifest.model_validate_json(json.dumps(finalization_chain_payload["attempt"]))


@pytest.fixture
def attempt_readiness(finalization_chain_payload: dict[str, object]) -> AttemptReadiness:
    return AttemptReadiness.model_validate_json(json.dumps(finalization_chain_payload["readiness"]))


@pytest.fixture
def finalization_client_result(finalization_chain_payload: dict[str, object]) -> ClientResult:
    return ClientResult.model_validate_json(json.dumps(finalization_chain_payload["client_result"]))


@pytest.fixture
def candidate_output_manifest(finalization_chain_payload: dict[str, object]) -> CandidateOutputManifest:
    return CandidateOutputManifest.model_validate_json(json.dumps(finalization_chain_payload["candidate"]))


@pytest.fixture
def shard_winner(finalization_chain_payload: dict[str, object]) -> ShardWinner:
    return ShardWinner.model_validate_json(json.dumps(finalization_chain_payload["winner"]))


@pytest.fixture
def fake_clock() -> FakeClock:
    """Return an isolated explicitly controlled clock."""
    return FakeClock(datetime(2026, 8, 18, 12, tzinfo=UTC), monotonic_time=100.0)


@pytest.fixture
def fake_slurm_runner() -> FakeSlurmRunner:
    """Return an isolated Slurm runner with one array and one bounded sinfo query."""
    return FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(
                tasks=(
                    FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),
                    FakeSlurmTask(
                        SchedulerIdentity(array_job_id=4101, array_task_id=1),
                        queue_state="RUNNING",
                    ),
                )
            ),
        ),
        sinfo_responses={
            ("sinfo", "--noheader", "--format=%P|%G"): FakeCommandResponse(
                stdout=(SLURM_GOLDEN_DIRECTORY / "sinfo_gres.txt").read_text()
            )
        },
    )


@pytest.fixture
def fake_plugin_overlay() -> Path:
    """Return the installed-layout fake Data Designer plugin overlay."""
    return TEST_DIRECTORY / "fixtures" / "fake_plugin_overlay"


@pytest.fixture
def fake_logical_endpoint() -> FakeLogicalEndpoint:
    """Return an isolated two-backend logical endpoint."""
    return FakeLogicalEndpoint(
        "http://127.0.0.1:31000",
        (
            FakeVllmBackend("http://127.0.0.1:31001", rank=0),
            FakeVllmBackend("http://127.0.0.1:31002", rank=1),
        ),
    )
