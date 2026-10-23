# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest
from conftest import ClientWorkerCase, FakeDataDesigner
from pytest_httpx import HTTPXMock

from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.execution import ClientWorker
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import compute_serialized_json_sha256
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.resolution import resolve_slurm_config


@pytest.fixture
def external_client_case(
    external_run: DataDesignerSlurmConfig,
    client_worker_case: ClientWorkerCase,
) -> ClientWorkerCase:
    case = client_worker_case
    plan = SlurmRunCompiler.compile(
        resolve_slurm_config(
            external_run,
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
    monkeypatch.setenv("EXTERNAL_API_KEY", "fake-provider-secret")
    monkeypatch.setenv("EXTERNAL_HEADER_TOKEN", "fake-header-secret")
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
    assert providers["external"].api_key == "fake-provider-secret"
    assert providers["external"].extra_body == {"routing": {"version": 2}}
    assert providers["external"].extra_headers == {"X-Project": "test", "X-Token": "fake-header-secret"}
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


@pytest.mark.parametrize("missing", ("EXTERNAL_API_KEY", "EXTERNAL_HEADER_TOKEN"))
def test_worker_rejects_unavailable_hosted_credentials(
    external_client_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setenv("EXTERNAL_API_KEY", "fake-provider-secret")
    monkeypatch.setenv("EXTERNAL_HEADER_TOKEN", "fake-header-secret")
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
    monkeypatch.setenv("EXTERNAL_API_KEY", "fake-provider-secret")
    monkeypatch.setenv("EXTERNAL_HEADER_TOKEN", "fake-header-secret")
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


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_native_dd_generates_through_managed_and_external_endpoints(
    external_client_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock: HTTPXMock,
) -> None:
    case = external_client_case
    monkeypatch.setattr("data_designer.engine.models.telemetry.TELEMETRY_ENABLED", False)
    monkeypatch.setenv("EXTERNAL_API_KEY", "fake-provider-secret")
    monkeypatch.setenv("EXTERNAL_HEADER_TOKEN", "fake-header-secret")
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
    result = worker.run(case.plan_path, prepared=case.prepared, endpoints=case.endpoints, plugins=())
    assert result.actual_records == plan.shards[0].requested_records
    table = pq.read_table(case.attempt_dir / "dataset/parquet-files")
    assert table["answer"].to_pylist() == ["hello"] * result.actual_records
    requests = httpx_mock.get_requests(url=hosted_url)
    assert requests
    assert all(request.headers["Authorization"] == "Bearer fake-provider-secret" for request in requests)
    assert all(request.headers["X-Token"] == "fake-header-secret" for request in requests)
    assert all(json.loads(request.content)["model"] == "example/hosted-judge" for request in requests)
    for endpoint in case.endpoints.values():
        local_requests = httpx_mock.get_requests(url=endpoint + "/chat/completions")
        assert local_requests
        assert all("fake-provider-secret" not in str(request.headers) for request in local_requests)
        assert all("fake-header-secret" not in str(request.headers) for request in local_requests)
        assert all(json.loads(request.content)["model"] == "example/generator" for request in local_requests)
