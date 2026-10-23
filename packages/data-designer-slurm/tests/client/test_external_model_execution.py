# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from conftest import ClientWorkerCase, FakeDataDesigner
from pytest_httpx import HTTPXMock

from data_designer.config import DataDesignerConfigBuilder
from data_designer.config.models import ModelProvider
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.execution import ClientWorker
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import compute_serialized_json_sha256
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.state.artifacts import CandidateArtifactVerifier
from data_designer.slurm.state.outputs import CandidateOutputManifest


@pytest.fixture
def external_client_case(
    hosted_model_providers: tuple[ModelProvider, ...],
    external_run: DataDesignerSlurmConfig,
    client_worker_case: ClientWorkerCase,
) -> ClientWorkerCase:
    case = client_worker_case
    plan = SlurmRunCompiler.compile(
        resolve_slurm_config(
            external_run,
            hosted_model_providers=hosted_model_providers,
            selected_profile=case.plan.selected_profile,
            client_runtime=case.plan.client.runtime,
            deployment_images=tuple(case.plan.deployments[0].image for _ in external_run.deployments),
            dependency_lock=case.lock,
            runtime_bundle=case.plan.runtime_bundle,
            run_id=case.plan.run_id,
            package_version=case.plan.package_version,
        )
    )
    case.plan_path.write_text(plan.serialize_json())
    return ClientWorkerCase(
        plan,
        case.plan_path,
        case.lock,
        case.attempt_dir,
        case.scratch_root,
        case.prepared,
        case.endpoints if external_run.deployments else {},
    )


def test_worker_preserves_hosted_models_and_scopes_secrets(
    external_client_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = external_client_case
    monkeypatch.setenv("MODEL_API_KEY", "fake-provider-secret")
    designers: list[FakeDataDesigner] = []

    def factory(**kwargs: object) -> FakeDataDesigner:
        designer = FakeDataDesigner(**kwargs)
        designers.append(designer)
        return designer

    worker = ClientWorker(data_designer_factory=factory)
    worker.preflight(case.plan_path, prepared=case.prepared, endpoints=case.endpoints, plugins=())
    model = next(config for config in designers[0].validated_builder.model_configs if config.alias == "judge")
    assert model.model == "example/hosted-judge"
    assert model.provider == "external"
    assert model.inference_parameters.temperature == 0.3
    assert model.inference_parameters.extra_body == {"reasoning_effort": "medium"}
    assert model.inference_parameters.max_parallel_requests == 3
    providers = {provider.name: provider for provider in designers[0].initialization["model_providers"]}
    assert providers["external"].endpoint == "https://models.example/v1"
    assert providers["external"].api_key == "MODEL_API_KEY"
    assert providers["external"].extra_body == {"routing": {"version": 2}}
    assert providers["external"].extra_headers == {"X-Project": "test"}
    if case.plan.deployments:
        assert providers["slurm-generator"].endpoint == case.endpoints["generator"]
    else:
        assert set(providers) == {"external"}
    result = worker.run(case.plan_path, prepared=case.prepared, endpoints=case.endpoints, plugins=())
    assert result.actual_records == case.plan.shards[0].requested_records
    for path in case.plan_path.parent.rglob("*.json"):
        content = path.read_text()
        assert "fake-provider-secret" not in content
        assert "fake-header-secret" not in content


@pytest.mark.parametrize("missing", ("MODEL_API_KEY",))
def test_worker_rejects_unavailable_hosted_credentials(
    external_client_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "fake-provider-secret")
    monkeypatch.delenv(missing)
    case = external_client_case
    with pytest.raises(ClientWorkerError, match="required client secret is unavailable"):
        ClientWorker(data_designer_factory=FakeDataDesigner).preflight(
            case.plan_path,
            prepared=case.prepared,
            endpoints=case.endpoints,
            plugins=(),
        )


@pytest.mark.parametrize("external_run", ("hosted-only",), indirect=True)
def test_hosted_only_worker_needs_no_scheduler_gpu_environment(
    external_client_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MODEL_API_KEY", "fake-provider-secret")
    case = external_client_case
    worker = ClientWorker(data_designer_factory=FakeDataDesigner, environment={})
    worker.preflight(case.plan_path, prepared=case.prepared, endpoints={}, plugins=())
    result = worker.run(
        case.plan_path,
        prepared=case.prepared,
        endpoints={},
        plugins=(),
    )
    assert result.actual_records == case.plan.shards[0].requested_records


@pytest.mark.parametrize("existing_kind", ("nonempty", "symlink", "dangling-symlink", "raced"))
def test_worker_preserves_existing_dataset_paths(
    external_client_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    case = external_client_case
    monkeypatch.setenv("MODEL_API_KEY", "fake-provider-secret")
    worker = ClientWorker(data_designer_factory=FakeDataDesigner)
    worker.preflight(case.plan_path, prepared=case.prepared, endpoints=case.endpoints, plugins=())
    dataset = case.attempt_dir / "dataset"
    if existing_kind in {"nonempty", "raced"}:
        dataset.mkdir()
        if existing_kind == "nonempty":
            (dataset / "retained.txt").write_text("retain")
        else:
            rmdir = type(dataset).rmdir

            def racing_rmdir(path: Path) -> None:
                if path == dataset:
                    (dataset / "retained.txt").write_text("retain")
                rmdir(path)

            monkeypatch.setattr(type(dataset), "rmdir", racing_rmdir)
    else:
        target = case.plan_path.parent / "unrelated"
        if existing_kind == "symlink":
            target.mkdir()
        dataset.symlink_to(target, target_is_directory=True)
    message = "unavailable" if existing_kind == "raced" else "not empty"
    with pytest.raises(ClientWorkerError, match=f"attempt dataset workspace is {message}"):
        worker.run(case.plan_path, prepared=case.prepared, endpoints=case.endpoints, plugins=())
    if existing_kind in {"nonempty", "raced"}:
        assert (dataset / "retained.txt").read_text() == "retain"
    else:
        assert dataset.is_symlink()
        assert not target.exists() or list(target.iterdir()) == []


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.parametrize("seeded", (False, True))
def test_native_dd_generates_through_managed_and_external_endpoints(
    external_client_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: HTTPXMock,
    seeded: bool,
) -> None:
    case = external_client_case
    monkeypatch.setattr("data_designer.engine.models.telemetry.TELEMETRY_ENABLED", False)
    monkeypatch.setenv("MODEL_API_KEY", "fake-provider-secret")
    payload = case.plan.model_dump(mode="python")
    payload["builder"]["inline"]["data_designer"]["columns"] = [
        {
            "column_type": "llm-text",
            "name": "answer",
            "model_alias": "judge",
            "prompt": "Review {{ draft }}." if case.plan.deployments else "Say hello.",
        }
    ]
    if case.plan.deployments:
        payload["builder"]["inline"]["data_designer"]["columns"].insert(
            0,
            {
                "column_type": "llm-text",
                "name": "draft",
                "model_alias": "generator",
                "prompt": "Say hello.",
            },
        )
    if seeded:
        seed_path = case.plan_path.parent / "seed.jsonl"
        seed_path.write_text("".join(json.dumps({"seed_id": index}) + "\n" for index in range(8)))
        builder = DataDesignerConfigBuilder.from_config(payload["builder"]["inline"])
        builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)), sampling_strategy="ordered")
        payload["builder"]["inline"]["data_designer"]["seed_config"] = builder.get_seed_config().model_dump(mode="json")
    payload["builder"]["content_sha256"] = compute_serialized_json_sha256(payload["builder"]["inline"])
    plan = ResolvedSlurmRunPlan.model_validate(payload)
    case.plan_path.write_text(plan.serialize_json())
    response = {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "example/hosted-judge",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    hosted_url = "https://models.example/v1/chat/completions"
    httpx_mock.add_response(url=hosted_url, json=response)
    for endpoint in case.endpoints.values():
        httpx_mock.add_response(url=endpoint + "/chat/completions", json=response)
    worker = ClientWorker(environment={"SLURM_JOB_GPUS": "0"} if case.plan.deployments else {})
    worker.preflight(case.plan_path, prepared=case.prepared, endpoints=case.endpoints, plugins=())
    # The allocation controller claims the dataset directory before running DD.
    (case.attempt_dir / "dataset").mkdir(exist_ok=True)
    result = worker.run(case.plan_path, prepared=case.prepared, endpoints=case.endpoints, plugins=())
    assert result.actual_records == plan.shards[0].requested_records
    assert result.candidate_output_manifest is not None
    candidate = CandidateOutputManifest.model_validate_json(Path(result.candidate_output_manifest.path).read_text())
    with CandidateArtifactVerifier().verify(candidate) as artifacts:
        assert sum(artifacts.record_counts) == result.actual_records
        artifacts.rebind()
    table = pq.read_table(case.attempt_dir / "dataset/parquet-files")
    assert table["answer"].to_pylist() == ["hello"] * result.actual_records
    if seeded:
        assert sorted(table["seed_id"].to_pylist()) == list(range(8))
    requests = httpx_mock.get_requests(url=hosted_url)
    assert requests
    assert all(request.headers["Authorization"] == "Bearer fake-provider-secret" for request in requests)
    assert all(request.headers["X-Project"] == "test" for request in requests)
    assert all(json.loads(request.content)["model"] == "example/hosted-judge" for request in requests)
    for endpoint in case.endpoints.values():
        local_requests = httpx_mock.get_requests(url=endpoint + "/chat/completions")
        assert local_requests
        assert all("fake-provider-secret" not in str(request.headers) for request in local_requests)
        assert all("fake-header-secret" not in str(request.headers) for request in local_requests)
        assert all(json.loads(request.content)["model"] == "example/generator" for request in local_requests)
