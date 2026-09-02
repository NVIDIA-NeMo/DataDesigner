# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from packaging.tags import interpreter_name, interpreter_version

from data_designer.config import ResumeMode
from data_designer.slurm.client.environment import PreparedClientEnvironment
from data_designer.slurm.client.records import ClientInstallerOutcome
from data_designer.slurm.contracts import InstalledDistribution, compute_canonical_json_sha256
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan

GOLDEN_DIRECTORY = Path(__file__).parents[1] / "contracts" / "golden"


@dataclass(frozen=True)
class ClientWorkerCase:
    plan: ResolvedSlurmRunPlan
    plan_path: Path
    lock: ResolvedDependencyLock
    attempt_dir: Path
    prepared: PreparedClientEnvironment
    endpoints: dict[str, str]


class FakeCreationResults:
    def __init__(
        self,
        dataset_path: Path,
        *,
        requested: int,
        actual: int,
        resume: ResumeMode,
    ) -> None:
        self.dataset_path = dataset_path
        self.requested_num_records = requested
        self.actual_num_records = actual
        self.early_shutdown = actual < requested
        self.requested_resume_mode = resume
        self.effective_resume_mode = ResumeMode.NEVER

    def export(self, path: Path, *, format: str) -> Path:
        assert format == "parquet"
        pq.write_table(pa.table({"value": tuple(range(self.actual_num_records))}), path)
        return path


class FakeDataDesigner:
    def __init__(
        self,
        *,
        actual_records: int | None = None,
        create_error: BaseException | None = None,
        create_error_after_batch: BaseException | None = None,
        effective_resume: ResumeMode = ResumeMode.NEVER,
        **kwargs: object,
    ) -> None:
        self.actual_records = actual_records
        self.create_error = create_error
        self.create_error_after_batch = create_error_after_batch
        self.effective_resume = effective_resume
        self.initialization = kwargs
        self.run_config = None
        self.validated_builder = None

    def set_run_config(self, run_config: object) -> None:
        self.run_config = run_config

    def validate(self, config_builder: object) -> None:
        self.validated_builder = config_builder

    def create(
        self,
        config_builder: object,
        *,
        num_records: int,
        dataset_name: str,
        resume: ResumeMode,
        artifact_path: Path,
        on_batch_complete: object,
    ) -> FakeCreationResults:
        del config_builder
        if self.create_error is not None:
            raise self.create_error
        actual = num_records if self.actual_records is None else self.actual_records
        dataset_path = artifact_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        batch_path = dataset_path / "batch.parquet"
        pq.write_table(pa.table({"value": tuple(range(actual))}), batch_path)
        on_batch_complete(batch_path)
        if self.create_error_after_batch is not None:
            raise self.create_error_after_batch
        results = FakeCreationResults(dataset_path, requested=num_records, actual=actual, resume=resume)
        results.effective_resume_mode = self.effective_resume
        return results


@pytest.fixture
def client_worker_case(tmp_path: Path) -> ClientWorkerCase:
    workspace = tmp_path / "workspace"
    payload = json.loads(
        (GOLDEN_DIRECTORY / "single_node_plan.json").read_text().replace("/workspace/primary", workspace.as_posix())
    )
    python_abi = f"{interpreter_name()}{interpreter_version()}"
    payload["client"]["image"]["inspection"]["inspection"].update(
        {"python_abi": python_abi, "python_version": platform.python_version()}
    )
    lock_payload = json.loads((GOLDEN_DIRECTORY / "dependency_lock_single.json").read_text())
    lock_payload["python_abi"] = python_abi
    lock = ResolvedDependencyLock.model_validate_json(json.dumps(lock_payload))
    payload["client"]["dependency_lock"]["sha256"] = lock.compute_sha256()
    payload["selected_profile"]["profile_sha256"] = compute_canonical_json_sha256(
        payload["selected_profile"]["profile"]
    )
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    plan_path = workspace / "runs" / plan.run_id / "resolved-plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(plan.serialize_json())
    lock_path = Path(plan.client.dependency_lock.path)
    lock_path.write_text(lock.serialize_json())
    Path(plan.invocation.effective_input_bindings.managed_assets_path).mkdir(parents=True)
    shard = plan.shards[0]
    attempt_dir = Path(shard.resume_workspace.path).parent / "attempts" / "attempt-0001"
    overlay_path = attempt_dir / "client-env" / "site-packages"
    overlay_path.mkdir(parents=True)
    prepared = PreparedClientEnvironment(
        run_id=plan.run_id,
        shard_id=shard.shard_id,
        attempt_id="attempt-0001",
        attempt_dir=attempt_dir,
        overlay_path=overlay_path,
        dependency_lock=plan.client.dependency_lock,
        client_image_sha256=plan.client.image.sha256,
        python_abi=lock.python_abi,
        installer_outcome=ClientInstallerOutcome.NOT_REQUIRED,
        installed_distributions=tuple(
            InstalledDistribution(name=distribution.name, version=distribution.version)
            for distribution in lock.image_distributions
        ),
    )
    endpoints = {plan.deployments[0].authored.model_alias: f"http://127.0.0.1:{plan.client.ports[0].port}/v1"}
    return ClientWorkerCase(plan, plan_path, lock, attempt_dir, prepared, endpoints)
