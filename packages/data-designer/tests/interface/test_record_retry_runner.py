# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import data_designer.interface.record_retry_attempts as record_retry_attempts_module
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.sampler_params import UUIDSamplerParams
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.record_retry import RetryExhaustion, RetryUntil, SamplerRetryMode
from data_designer.interface.record_retry_attempts import RecordRetryAttemptRunner
from data_designer.interface.record_retry_builders import RecordRetryBuilderFactory
from data_designer.interface.record_retry_publisher import RecordRetryPublisher
from data_designer.interface.record_retry_runner import RecordRetryRunner
from data_designer.interface.record_retry_state import (
    ACCEPTED_DIRECTORY,
    ATTEMPT_COMPLETION_FILENAME,
    ATTEMPTS_DIRECTORY,
    BASE_COHORT_PATH,
    FINAL_COMPLETION_FILENAME,
    INTERNAL_ATTEMPT_COLUMN_BASENAME,
    INTERNAL_SLOT_COLUMN_BASENAME,
    MANIFEST_FILENAME,
    AttemptManifest,
    RetryManifest,
    get_attempt_accepted_path,
    load_retry_manifest,
    read_attempt_completion,
)
from data_designer.interface.record_retry_utils import (
    aggregate_model_usage,
    classify_attempt,
    clear_ambiguous_finalization,
    copy_preserved_seed_media,
    load_and_validate_base_cohort,
    load_committed_accepted_ids,
    normalize_slot_ids,
    package_accepted_media,
    read_accepted_slot_ids,
    strict_predicate_outcome,
    write_or_validate_attempt_input,
)


def _real_data_designer(
    artifact_path: Path,
    model_providers: list[ModelProvider],
) -> DataDesigner:
    return DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
        secret_resolver=PlaintextResolver(),
    )


def _retry_builder(
    model_configs: list[ModelConfig],
    *,
    acceptance_attempts: dict[int, int],
    call_counts: dict[int, int],
    sampler_history: dict[int, list[str]] | None = None,
) -> DataDesignerConfigBuilder:
    required_columns = ["seed_id"]
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"seed_id": list(acceptance_attempts)})),
        sampling_strategy=SamplingStrategy.ORDERED,
    )
    if sampler_history is not None:
        builder.add_column(
            SamplerColumnConfig(
                name="sample_id",
                sampler_type="uuid",
                params=UUIDSamplerParams(),
            )
        )
        required_columns.append("sample_id")

    @custom_column_generator(required_columns=required_columns, side_effect_columns=["accepted"])
    def generate(row: dict[str, Any]) -> dict[str, Any]:
        seed_id = int(row["seed_id"])
        call_counts[seed_id] = call_counts.get(seed_id, 0) + 1
        if sampler_history is not None:
            sampler_history.setdefault(seed_id, []).append(str(row["sample_id"]))
        row["generated"] = f"seed-{seed_id}-attempt-{call_counts[seed_id]}"
        row["accepted"] = call_counts[seed_id] >= acceptance_attempts[seed_id]
        return row

    builder.add_column(CustomColumnConfig(name="generated", generator_function=generate))
    builder.add_processor(DropColumnsProcessorConfig(name="drop-predicate", column_names=["accepted"]))
    return builder


def _manifest(stage_path: Path) -> dict[str, Any]:
    return json.loads((stage_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _workflow_metadata(artifact_path: Path, workflow_name: str) -> dict[str, Any]:
    return json.loads((artifact_path / workflow_name / "workflow-metadata.json").read_text(encoding="utf-8"))


def _attempt_payload(accepted_records: int = 1) -> dict[str, Any]:
    return {
        "input_records": 1,
        "output_records": 1,
        "accepted_records": accepted_records,
        "false_records": 1 - accepted_records,
        "null_records": 0,
        "missing_records": 0,
    }


def _usage_payload(input_tokens: int) -> dict[str, Any]:
    return {
        "token_usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "request_usage": {"successful_requests": 1, "failed_requests": 0},
        "tool_usage": {},
        "image_usage": {},
    }


def _completed_single_record_run(
    tmp_path: Path,
    model_configs: list[ModelConfig],
    model_providers: list[ModelProvider],
) -> tuple[DataDesigner, DataDesignerConfigBuilder, RetryUntil, Path]:
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    data_designer = _real_data_designer(artifact_path, model_providers)
    builder = _retry_builder(
        model_configs,
        acceptance_attempts={0: 1},
        call_counts={},
    )
    policy = RetryUntil(predicate_column="accepted", max_attempts=1)
    RecordRetryRunner(data_designer).run(
        config_builder=builder,
        num_records=1,
        dataset_name="records",
        artifact_path=artifact_path,
        policy=policy,
        fingerprint="stable-fingerprint",
        resume=ResumeMode.NEVER,
        workflow_resume=ResumeMode.NEVER,
    )
    stage_path = artifact_path / "records"
    assert (stage_path / FINAL_COMPLETION_FILENAME).is_file()
    return data_designer, builder, policy, stage_path


@pytest.mark.parametrize("sampler_mode", [SamplerRetryMode.PRESERVE, SamplerRetryMode.RESAMPLE])
def test_real_workflow_retries_only_pending_slots_and_honors_sampler_mode(
    sampler_mode: SamplerRetryMode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    data_designer = _real_data_designer(artifact_path, stub_model_providers)
    call_counts: dict[int, int] = {}
    sampler_history: dict[int, list[str]] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 1, 1: 2, 2: 3},
        call_counts=call_counts,
        sampler_history=sampler_history,
    )

    original_create = data_designer._create
    create_processor_names: list[list[str]] = []

    def recording_create(config_builder: DataDesignerConfigBuilder, **kwargs: Any):
        create_processor_names.append([processor.name for processor in config_builder.get_processor_configs()])
        return original_create(config_builder, **kwargs)

    monkeypatch.setattr(data_designer, "_create", recording_create)
    workflow = data_designer.compose_workflow(name=f"retry-{sampler_mode.value}")
    workflow.add_stage(
        "records",
        builder,
        num_records=3,
        retry_until=RetryUntil(
            predicate_column="accepted",
            max_attempts=3,
            sampler_retry_mode=sampler_mode,
        ),
    )

    result = workflow.run()["records"]
    output = result.load_dataset().sort_values("seed_id").reset_index(drop=True)

    assert call_counts == {0: 1, 1: 2, 2: 3}
    assert output["seed_id"].tolist() == [0, 1, 2]
    assert output["generated"].tolist() == [
        "seed-0-attempt-1",
        "seed-1-attempt-2",
        "seed-2-attempt-3",
    ]
    assert "accepted" not in output
    assert not any(column.startswith("_data_designer_record_retry_") for column in output)
    assert sum("drop-predicate" in names for names in create_processor_names) == 1

    for seed_id, expected_calls in {0: 1, 1: 2, 2: 3}.items():
        history = sampler_history[seed_id]
        assert len(history) == expected_calls
        expected_unique = 1 if sampler_mode == SamplerRetryMode.PRESERVE else expected_calls
        assert len(set(history)) == expected_unique

    stage_path = artifact_path / f"retry-{sampler_mode.value}" / "stage-0-records"
    manifest = _manifest(stage_path)
    assert [attempt["input_records"] for attempt in manifest["attempts"]] == [3, 2, 1]
    assert [attempt["accepted_records"] for attempt in manifest["attempts"]] == [1, 1, 1]
    summary = _workflow_metadata(artifact_path, f"retry-{sampler_mode.value}")["stages"][0]["retry_summary"]
    assert summary == {
        "target_records": 3,
        "accepted_records": 3,
        "unresolved_records": 0,
        "unresolved_slot_ids": [],
        "candidate_records": 6,
        "attempts": 3,
        "sampler_retry_mode": sampler_mode.value,
        "exhausted": False,
        "distribution_warning": None,
    }


def test_seedless_cohort_uses_direct_slots_without_a_bootstrap_run(
    tmp_path: Path,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    @custom_column_generator(required_columns=["sample_id"], side_effect_columns=["accepted"])
    def generate(row: dict[str, Any]) -> dict[str, Any]:
        row["value"] = row["sample_id"]
        row["accepted"] = True
        return row

    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(
        SamplerColumnConfig(
            name="sample_id",
            sampler_type="uuid",
            params=UUIDSamplerParams(),
        )
    )
    builder.add_column(CustomColumnConfig(name="value", generator_function=generate))
    artifact_path = tmp_path / "artifacts"

    result = RecordRetryRunner(_real_data_designer(artifact_path, stub_model_providers)).run(
        config_builder=builder,
        num_records=3,
        dataset_name="records",
        artifact_path=artifact_path,
        policy=RetryUntil(
            predicate_column="accepted",
            max_attempts=1,
            sampler_retry_mode=SamplerRetryMode.RESAMPLE,
        ),
        fingerprint="fingerprint",
        resume=ResumeMode.NEVER,
        workflow_resume=ResumeMode.NEVER,
    )

    stage_path = artifact_path / "records"
    base = lazy.pd.read_parquet(stage_path / BASE_COHORT_PATH)
    assert len(base) == 3
    assert list(base.columns) == [_manifest(stage_path)["slot_column"]]
    assert not (stage_path / BASE_COHORT_PATH.parent / "run").exists()
    output = result.load_dataset()
    assert output["value"].tolist() == output["sample_id"].tolist()
    assert output["sample_id"].nunique() == 3


@pytest.mark.parametrize(
    ("acceptance_attempts", "on_exhausted", "expected_accepted"),
    [
        ({0: 1, 1: 1}, RetryExhaustion.RAISE, 2),
        ({0: 1, 1: 99, 2: 99}, RetryExhaustion.RETURN_PARTIAL, 1),
    ],
    ids=["full", "partial"],
)
def test_full_and_partial_publications_use_original_fingerprint_and_logical_target(
    acceptance_attempts: dict[int, int],
    on_exhausted: RetryExhaustion,
    expected_accepted: int,
    tmp_path: Path,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts=acceptance_attempts,
        call_counts=call_counts,
    )
    target_records = len(acceptance_attempts)
    policy = RetryUntil(
        predicate_column="accepted",
        max_attempts=1,
        on_exhausted=on_exhausted,
    )

    result = RecordRetryRunner(_real_data_designer(artifact_path, stub_model_providers)).run(
        config_builder=builder,
        num_records=target_records,
        dataset_name="records",
        artifact_path=artifact_path,
        policy=policy,
        fingerprint="workflow-stage-fingerprint",
        resume=ResumeMode.NEVER,
        workflow_resume=ResumeMode.NEVER,
    )

    metadata = result.artifact_storage.read_metadata()
    original_fingerprint = builder.build().fingerprint()
    assert {key: metadata[key] for key in original_fingerprint} == original_fingerprint
    assert metadata["target_num_records"] == target_records
    assert metadata["original_target_num_records"] == target_records
    assert metadata["actual_num_records"] == expected_accepted
    assert result.count_records() == expected_accepted
    analysis = result.load_analysis()
    assert analysis.target_num_records == target_records
    assert analysis.num_records == expected_accepted


def test_internal_columns_do_not_collide_with_declared_side_effect_columns(
    tmp_path: Path,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.with_seed_dataset(DataFrameSeedSource(df=lazy.pd.DataFrame({"seed_id": [0]})))

    @custom_column_generator(
        required_columns=["seed_id"],
        side_effect_columns=[
            "accepted",
            INTERNAL_SLOT_COLUMN_BASENAME,
            INTERNAL_ATTEMPT_COLUMN_BASENAME,
        ],
    )
    def generate(row: dict[str, Any]) -> dict[str, Any]:
        row["generated"] = "value"
        row["accepted"] = True
        row[INTERNAL_SLOT_COLUMN_BASENAME] = "user-slot"
        row[INTERNAL_ATTEMPT_COLUMN_BASENAME] = "user-attempt"
        return row

    builder.add_column(CustomColumnConfig(name="generated", generator_function=generate))
    workflow = _real_data_designer(artifact_path, stub_model_providers).compose_workflow(name="name-collision")
    workflow.add_stage(
        "records",
        builder,
        num_records=1,
        retry_until=RetryUntil(predicate_column="accepted", max_attempts=1),
    )

    output = workflow.run()["records"].load_dataset()

    assert output.loc[0, INTERNAL_SLOT_COLUMN_BASENAME] == "user-slot"
    assert output.loc[0, INTERNAL_ATTEMPT_COLUMN_BASENAME] == "user-attempt"
    manifest = _manifest(artifact_path / "name-collision" / "stage-0-records")
    assert manifest["slot_column"] == f"{INTERNAL_SLOT_COLUMN_BASENAME}_1"
    assert manifest["attempt_column"] == f"{INTERNAL_ATTEMPT_COLUMN_BASENAME}_1"
    assert manifest["slot_column"] not in output
    assert manifest["attempt_column"] not in output


@pytest.mark.parametrize("on_exhausted", [RetryExhaustion.RAISE, RetryExhaustion.RETURN_PARTIAL])
def test_real_workflow_exhaustion_raise_and_all_zero_partial(
    on_exhausted: RetryExhaustion,
    tmp_path: Path,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 99, 1: 99},
        call_counts=call_counts,
    )
    workflow = _real_data_designer(artifact_path, stub_model_providers).compose_workflow(
        name=f"exhaust-{on_exhausted.value}"
    )
    workflow.add_stage(
        "records",
        builder,
        num_records=2,
        retry_until=RetryUntil(
            predicate_column="accepted",
            max_attempts=2,
            on_exhausted=on_exhausted,
        ),
    )

    if on_exhausted == RetryExhaustion.RAISE:
        with pytest.raises(
            DataDesignerWorkflowError,
            match=r"exhausted after 2 attempt\(s\): accepted 0 of 2 slots after 4 candidate records",
        ):
            workflow.run()
    else:
        result = workflow.run()["records"]
        assert result.count_records() == 0
        assert result.load_dataset().empty
        metadata = _workflow_metadata(artifact_path, f"exhaust-{on_exhausted.value}")
        assert metadata["stages"][0]["status"] == "completed_empty"
        assert metadata["stages"][0]["retry_summary"] == {
            "target_records": 2,
            "accepted_records": 0,
            "unresolved_records": 2,
            "unresolved_slot_ids": [0, 1],
            "candidate_records": 4,
            "attempts": 2,
            "sampler_retry_mode": "preserve",
            "exhausted": True,
            "distribution_warning": (
                "The partial result omits unresolved seed/sampler slots and is biased toward cohort "
                "combinations that passed within the retry bounds."
            ),
        }

    assert call_counts == {0: 2, 1: 2}
    stage_path = artifact_path / f"exhaust-{on_exhausted.value}" / "stage-0-records"
    manifest = _manifest(stage_path)
    assert manifest["status"] == ("exhausted" if on_exhausted == RetryExhaustion.RAISE else "complete")
    assert [attempt["false_records"] for attempt in manifest["attempts"]] == [2, 2]
    assert sum(attempt["input_records"] for attempt in manifest["attempts"]) == 4


def test_candidate_budget_stops_before_overshoot_and_returns_nonempty_partial(
    tmp_path: Path,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 1, 1: 99, 2: 99},
        call_counts=call_counts,
    )
    workflow = _real_data_designer(artifact_path, stub_model_providers).compose_workflow(name="candidate-budget")
    workflow.add_stage(
        "records",
        builder,
        num_records=3,
        retry_until=RetryUntil(
            predicate_column="accepted",
            max_attempts=10,
            max_candidate_records=4,
            on_exhausted=RetryExhaustion.RETURN_PARTIAL,
        ),
    )

    result = workflow.run()["records"]

    assert call_counts == {0: 1, 1: 1, 2: 1}
    assert result.load_dataset()["seed_id"].tolist() == [0]
    stage_path = artifact_path / "candidate-budget" / "stage-0-records"
    assert [attempt["input_records"] for attempt in _manifest(stage_path)["attempts"]] == [3]
    summary = _workflow_metadata(artifact_path, "candidate-budget")["stages"][0]["retry_summary"]
    assert summary["candidate_records"] == 3
    assert summary["attempts"] == 1
    assert summary["accepted_records"] == 1
    assert summary["unresolved_slot_ids"] == [1, 2]
    assert summary["exhausted"] is True


def test_completed_empty_retry_stage_resumes_without_phantom_output_processor_result(
    tmp_path: Path,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 99},
        call_counts=call_counts,
    )
    boundary_processor = DropColumnsProcessorConfig(name="boundary-drop", column_names=["generated"])
    policy = RetryUntil(
        predicate_column="accepted",
        max_attempts=1,
        on_exhausted=RetryExhaustion.RETURN_PARTIAL,
    )
    data_designer = _real_data_designer(artifact_path, stub_model_providers)
    workflow = data_designer.compose_workflow(name="empty-resume")
    workflow.add_stage(
        "records",
        builder,
        num_records=1,
        retry_until=policy,
        output_processors=[boundary_processor],
    )

    first = workflow.run()["records"]

    resumed_workflow = data_designer.compose_workflow(name="empty-resume")
    resumed_workflow.add_stage(
        "records",
        builder,
        num_records=1,
        retry_until=policy,
        output_processors=[boundary_processor],
    )
    resumed = resumed_workflow.run(resume=ResumeMode.ALWAYS)["records"]
    stage_path = artifact_path / "empty-resume" / "stage-0-records"

    assert first.count_records() == 0
    assert resumed.count_records() == 0
    assert resumed.artifact_storage.base_dataset_path == stage_path
    assert not (stage_path / "output-processors").exists()
    assert call_counts == {0: 1}
    assert _workflow_metadata(artifact_path, "empty-resume")["stages"][0]["output_processor_output_path"] is None


def test_real_runner_resumes_committed_attempt_without_retrying_accepted_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 1, 1: 2},
        call_counts=call_counts,
    )
    policy = RetryUntil(predicate_column="accepted", max_attempts=2)
    data_designer = _real_data_designer(artifact_path, stub_model_providers)
    attempt_runner = RecordRetryAttemptRunner(data_designer)
    original_run_attempt = attempt_runner.run_attempt

    def interrupt_before_second_attempt(**kwargs: Any) -> AttemptManifest:
        if len(kwargs["manifest"].attempts) == 1:
            raise RuntimeError("simulated interruption")
        return original_run_attempt(**kwargs)

    monkeypatch.setattr(attempt_runner, "run_attempt", interrupt_before_second_attempt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        RecordRetryRunner(data_designer, attempt_runner=attempt_runner).run(
            config_builder=builder,
            num_records=2,
            dataset_name="records",
            artifact_path=artifact_path,
            policy=policy,
            fingerprint="stable-fingerprint",
            resume=ResumeMode.NEVER,
            workflow_resume=ResumeMode.NEVER,
        )

    stage_path = artifact_path / "records"
    assert call_counts == {0: 1, 1: 1}
    assert len(_manifest(stage_path)["attempts"]) == 1

    resumed = RecordRetryRunner(data_designer).run(
        config_builder=builder,
        num_records=2,
        dataset_name="records",
        artifact_path=artifact_path,
        policy=policy,
        fingerprint="stable-fingerprint",
        resume=ResumeMode.ALWAYS,
        workflow_resume=ResumeMode.ALWAYS,
    )

    assert call_counts == {0: 1, 1: 2}
    assert resumed.load_dataset().sort_values("seed_id")["generated"].tolist() == [
        "seed-0-attempt-1",
        "seed-1-attempt-2",
    ]
    assert resumed.artifact_storage.read_metadata()["record_retry"]["candidate_records"] == 3
    assert [attempt["input_records"] for attempt in _manifest(stage_path)["attempts"]] == [2, 1]


def test_real_runner_recovers_uncommitted_attempt_artifact_without_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 1, 1: 2},
        call_counts=call_counts,
    )
    policy = RetryUntil(predicate_column="accepted", max_attempts=2)
    data_designer = _real_data_designer(artifact_path, stub_model_providers)
    original_package_media = record_retry_attempts_module.package_accepted_media

    def interrupt_after_attempt_artifact(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("simulated packaging interruption")

    monkeypatch.setattr(record_retry_attempts_module, "package_accepted_media", interrupt_after_attempt_artifact)
    with pytest.raises(RuntimeError, match="simulated packaging interruption"):
        RecordRetryRunner(data_designer).run(
            config_builder=builder,
            num_records=2,
            dataset_name="records",
            artifact_path=artifact_path,
            policy=policy,
            fingerprint="stable-fingerprint",
            resume=ResumeMode.NEVER,
            workflow_resume=ResumeMode.NEVER,
        )

    stage_path = artifact_path / "records"
    assert call_counts == {0: 1, 1: 1}
    assert _manifest(stage_path)["attempts"] == []
    assert (stage_path / ATTEMPTS_DIRECTORY / "attempt-000" / "run" / "parquet-files").is_dir()
    assert (stage_path / ATTEMPTS_DIRECTORY / "attempt-000" / ATTEMPT_COMPLETION_FILENAME).is_file()
    assert not (stage_path / get_attempt_accepted_path(0)).exists()

    monkeypatch.setattr(record_retry_attempts_module, "package_accepted_media", original_package_media)
    resumed = RecordRetryRunner(data_designer).run(
        config_builder=builder,
        num_records=2,
        dataset_name="records",
        artifact_path=artifact_path,
        policy=policy,
        fingerprint="stable-fingerprint",
        resume=ResumeMode.ALWAYS,
        workflow_resume=ResumeMode.ALWAYS,
    )

    assert call_counts == {0: 1, 1: 2}
    assert resumed.load_dataset().sort_values("seed_id")["generated"].tolist() == [
        "seed-0-attempt-1",
        "seed-1-attempt-2",
    ]
    assert [attempt["input_records"] for attempt in _manifest(stage_path)["attempts"]] == [2, 1]


def test_real_runner_resumes_after_final_artifact_was_published_before_manifest_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 1},
        call_counts=call_counts,
    )
    policy = RetryUntil(predicate_column="accepted", max_attempts=1)
    data_designer = _real_data_designer(artifact_path, stub_model_providers)
    publisher = RecordRetryPublisher(data_designer)
    original_publish = publisher.publish

    def interrupt_after_finalization(**kwargs: Any) -> None:
        original_publish(**kwargs)
        raise RuntimeError("simulated post-publication interruption")

    monkeypatch.setattr(publisher, "publish", interrupt_after_finalization)
    with pytest.raises(RuntimeError, match="simulated post-publication interruption"):
        RecordRetryRunner(data_designer, publisher=publisher).run(
            config_builder=builder,
            num_records=1,
            dataset_name="records",
            artifact_path=artifact_path,
            policy=policy,
            fingerprint="stable-fingerprint",
            resume=ResumeMode.NEVER,
            workflow_resume=ResumeMode.NEVER,
        )

    stage_path = artifact_path / "records"
    assert _manifest(stage_path)["status"] == "finalizing"
    assert call_counts == {0: 1}

    resumed = RecordRetryRunner(data_designer).run(
        config_builder=builder,
        num_records=1,
        dataset_name="records",
        artifact_path=artifact_path,
        policy=policy,
        fingerprint="stable-fingerprint",
        resume=ResumeMode.ALWAYS,
        workflow_resume=ResumeMode.ALWAYS,
    )

    assert call_counts == {0: 1}
    assert resumed.load_dataset()["generated"].tolist() == ["seed-0-attempt-1"]
    assert _manifest(stage_path)["status"] == "complete"


def test_final_completion_resume_skips_generation_and_reprofiling_and_restores_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    data_designer, builder, policy, stage_path = _completed_single_record_run(
        tmp_path,
        stub_model_configs,
        stub_model_providers,
    )
    manifest = _manifest(stage_path)
    manifest["base_model_usage"] = {"model": _usage_payload(1)}
    manifest["attempts"][0]["model_usage"] = {"model": _usage_payload(2)}
    (stage_path / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    completion_path = stage_path / FINAL_COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["model_usage"] = {"model": _usage_payload(3)}
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    create = MagicMock(side_effect=AssertionError("durably completed final output must not rerun"))
    create_profiler = MagicMock(side_effect=AssertionError("durable resume must not reprofile"))
    monkeypatch.setattr(data_designer, "_create", create)
    monkeypatch.setattr(data_designer, "_create_dataset_profiler", create_profiler)

    resumed = RecordRetryRunner(data_designer).run(
        config_builder=builder,
        num_records=1,
        dataset_name="records",
        artifact_path=stage_path.parent,
        policy=policy,
        fingerprint="stable-fingerprint",
        resume=ResumeMode.ALWAYS,
        workflow_resume=ResumeMode.ALWAYS,
    )

    create.assert_not_called()
    create_profiler.assert_not_called()
    assert resumed.model_usage is not None
    assert resumed.model_usage["model"]["token_usage"]["input_tokens"] == 6
    assert resumed.model_usage["model"]["request_usage"]["successful_requests"] == 3
    analysis = resumed.load_analysis()
    assert analysis.num_records == 1
    assert analysis.target_num_records == 1
    metadata = resumed.artifact_storage.read_metadata()
    assert metadata["record_retry"]["model_usage"] == resumed.model_usage


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("corrupt", "Final completion marker .* is corrupt"),
        ("invalid", "Final completion marker .* is corrupt"),
        ("coalesced-count", "does not match the coalesced accepted-record count"),
        ("dataset-count", "Final dataset does not match its durable record-retry completion marker"),
    ],
)
def test_final_completion_resume_rejects_corrupt_or_count_mismatched_state(
    case: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    data_designer, builder, policy, stage_path = _completed_single_record_run(
        tmp_path,
        stub_model_configs,
        stub_model_providers,
    )
    completion_path = stage_path / FINAL_COMPLETION_FILENAME
    if case == "corrupt":
        completion_path.write_text("{not-json", encoding="utf-8")
    elif case == "invalid":
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["accepted_records"] = 0
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
    elif case == "coalesced-count":
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["accepted_records"] = 2
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
    else:
        batch_path = next((stage_path / "parquet-files").glob("batch_*.parquet"))
        output = lazy.pd.read_parquet(batch_path)
        lazy.pd.concat([output, output], ignore_index=True).to_parquet(batch_path, index=False)

    create = MagicMock(side_effect=AssertionError("invalid durable state must not rerun generation"))
    create_profiler = MagicMock(side_effect=AssertionError("invalid durable state must not reprofile"))
    monkeypatch.setattr(data_designer, "_create", create)
    monkeypatch.setattr(data_designer, "_create_dataset_profiler", create_profiler)

    with pytest.raises(DataDesignerWorkflowError, match=message):
        RecordRetryRunner(data_designer).run(
            config_builder=builder,
            num_records=1,
            dataset_name="records",
            artifact_path=stage_path.parent,
            policy=policy,
            fingerprint="stable-fingerprint",
            resume=ResumeMode.ALWAYS,
            workflow_resume=ResumeMode.ALWAYS,
        )

    create.assert_not_called()
    create_profiler.assert_not_called()


def test_zero_output_completion_marker_is_reclassified_without_regeneration(
    tmp_path: Path,
    stub_model_configs: list[ModelConfig],
) -> None:
    call_counts: dict[int, int] = {}
    builder = _retry_builder(
        stub_model_configs,
        acceptance_attempts={0: 99},
        call_counts=call_counts,
    )
    policy = RetryUntil(predicate_column="accepted", max_attempts=1)
    builder_factory = RecordRetryBuilderFactory(builder, policy)
    stage_path = tmp_path / "records"
    attempt_dir = stage_path / ATTEMPTS_DIRECTORY / "attempt-000"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / ATTEMPT_COMPLETION_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_slot_ids": [0],
                "output_records": 0,
                "model_usage": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=1,
        policy=policy.model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
    )
    data_designer = MagicMock()

    attempt = RecordRetryAttemptRunner(data_designer).run_attempt(
        builder_factory=builder_factory,
        stage_path=stage_path,
        manifest=manifest,
        base_df=lazy.pd.DataFrame({"slot": [0], "seed_id": [0]}),
        pending_ids=[0],
    )

    data_designer._create.assert_not_called()
    assert call_counts == {}
    assert attempt.input_records == 1
    assert attempt.output_records == 0
    assert attempt.accepted_records == 0
    assert attempt.false_records == 0
    assert attempt.null_records == 0
    assert attempt.missing_records == 1
    accepted = lazy.pd.read_parquet(stage_path / get_attempt_accepted_path(0))
    assert accepted.empty
    assert {"slot", "attempt", "seed_id", "generated", "accepted"}.issubset(accepted.columns)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{not-json", "is corrupt"),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "input_slot_ids": [0],
                    "output_records": 2,
                    "model_usage": {},
                }
            ),
            "is corrupt",
        ),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "input_slot_ids": [0, 0],
                    "output_records": 0,
                    "model_usage": {},
                }
            ),
            "is corrupt",
        ),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "input_slot_ids": [1],
                    "output_records": 0,
                    "model_usage": {},
                }
            ),
            "does not match the immutable attempt input",
        ),
    ],
)
def test_attempt_completion_marker_rejects_corruption_and_slot_mismatch(
    payload: str,
    message: str,
    tmp_path: Path,
) -> None:
    marker = tmp_path / ATTEMPT_COMPLETION_FILENAME
    marker.write_text(payload, encoding="utf-8")

    with pytest.raises(DataDesignerWorkflowError, match=message):
        read_attempt_completion(marker, [0])


def test_missing_manifest_with_durable_orphans_respects_workflow_resume_policy(tmp_path: Path) -> None:
    stage_path = tmp_path / "stage"
    (stage_path / ATTEMPTS_DIRECTORY / "attempt-000").mkdir(parents=True)
    (stage_path / ACCEPTED_DIRECTORY).mkdir()
    policy = RetryUntil(predicate_column="accepted", max_attempts=2)

    with pytest.raises(DataDesignerWorkflowError, match="retry manifest is missing"):
        load_retry_manifest(
            stage_path=stage_path,
            fingerprint="fingerprint",
            policy=policy,
            resume=ResumeMode.ALWAYS,
            workflow_resume=ResumeMode.ALWAYS,
        )

    assert (stage_path / ATTEMPTS_DIRECTORY).exists()
    assert (
        load_retry_manifest(
            stage_path=stage_path,
            fingerprint="fingerprint",
            policy=policy,
            resume=ResumeMode.ALWAYS,
            workflow_resume=ResumeMode.IF_POSSIBLE,
        )
        is None
    )
    assert stage_path.is_dir()
    assert list(stage_path.iterdir()) == []


def test_resume_discards_nondurable_or_corrupt_state_only_when_allowed(tmp_path: Path) -> None:
    policy = RetryUntil(predicate_column="accepted", max_attempts=1)

    nondurable = tmp_path / "nondurable"
    nondurable.mkdir()
    (nondurable / "stray.txt").write_text("not durable", encoding="utf-8")
    assert (
        load_retry_manifest(
            stage_path=nondurable,
            fingerprint="fingerprint",
            policy=policy,
            resume=ResumeMode.ALWAYS,
            workflow_resume=ResumeMode.ALWAYS,
        )
        is None
    )
    assert list(nondurable.iterdir()) == []

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / MANIFEST_FILENAME).write_text("{not-json", encoding="utf-8")
    assert (
        load_retry_manifest(
            stage_path=corrupt,
            fingerprint="fingerprint",
            policy=policy,
            resume=ResumeMode.ALWAYS,
            workflow_resume=ResumeMode.IF_POSSIBLE,
        )
        is None
    )
    assert list(corrupt.iterdir()) == []

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    manifest_payload = {
        "schema_version": 1,
        "fingerprint": "old-fingerprint",
        "target_records": 1,
        "policy": policy.model_dump(mode="json"),
        "slot_column": "slot",
        "attempt_column": "attempt",
    }
    (mismatched / MANIFEST_FILENAME).write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(DataDesignerWorkflowError, match="does not match the current stage fingerprint"):
        load_retry_manifest(
            stage_path=mismatched,
            fingerprint="new-fingerprint",
            policy=policy,
            resume=ResumeMode.ALWAYS,
            workflow_resume=ResumeMode.ALWAYS,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "base cohort is missing"),
        ("corrupt", "Cannot read base cohort"),
        ("shape", "base cohort shape is incompatible"),
        ("slots", "base cohort slot IDs are not contiguous"),
    ],
)
def test_resume_validates_base_cohort_before_reusing_attempts(case: str, message: str, tmp_path: Path) -> None:
    manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=2,
        policy=RetryUntil(predicate_column="accepted", max_attempts=1).model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
    )
    base_path = tmp_path / BASE_COHORT_PATH
    if case == "corrupt":
        base_path.parent.mkdir(parents=True)
        base_path.write_text("not parquet", encoding="utf-8")
    elif case == "shape":
        base_path.parent.mkdir(parents=True)
        lazy.pd.DataFrame({"slot": [0]}).to_parquet(base_path, index=False)
    elif case == "slots":
        base_path.parent.mkdir(parents=True)
        lazy.pd.DataFrame({"slot": [1, 0]}).to_parquet(base_path, index=False)

    with pytest.raises(DataDesignerWorkflowError, match=message):
        load_and_validate_base_cohort(tmp_path, manifest)


def test_classification_accounts_for_true_false_null_and_missing_rows() -> None:
    manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=4,
        policy=RetryUntil(predicate_column="accepted", max_attempts=1).model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
    )
    attempt_input = lazy.pd.DataFrame({"slot": [0, 1, 2, 3]})
    output = lazy.pd.DataFrame(
        {
            "slot": [0, 1, 2],
            "accepted": [True, False, None],
            "value": ["accepted", "false", "null"],
        }
    )

    accepted, counts = classify_attempt(
        output=output,
        attempt_input=attempt_input,
        manifest=manifest,
        predicate_column="accepted",
    )

    assert accepted[["slot", "value"]].to_dict(orient="records") == [{"slot": 0, "value": "accepted"}]
    assert counts == {
        "output_records": 3,
        "accepted_records": 1,
        "false_records": 1,
        "null_records": 1,
        "missing_records": 1,
    }


@pytest.mark.parametrize("column", ["seed_value", "preserved_sample"])
def test_classification_rejects_mutated_seed_or_preserved_sampler_values(column: str) -> None:
    policy = RetryUntil(predicate_column="accepted", max_attempts=1, sampler_retry_mode=SamplerRetryMode.PRESERVE)
    manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=1,
        policy=policy.model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
    )
    attempt_input = lazy.pd.DataFrame(
        {
            "slot": [0],
            "attempt": [0],
            "seed_value": ["seed-original"],
            "preserved_sample": ["sampler-original"],
        }
    )
    output = attempt_input.copy()
    output[column] = "mutated"
    output["accepted"] = True

    with pytest.raises(DataDesignerWorkflowError, match=rf"mutated stable seed/cohort column '{column}'"):
        classify_attempt(
            output=output,
            attempt_input=attempt_input,
            manifest=manifest,
            predicate_column="accepted",
        )


def test_classification_does_not_treat_resampled_values_as_stable_inputs() -> None:
    policy = RetryUntil(predicate_column="accepted", max_attempts=1, sampler_retry_mode=SamplerRetryMode.RESAMPLE)
    manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=1,
        policy=policy.model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
    )
    attempt_input = lazy.pd.DataFrame({"slot": [0], "attempt": [0], "seed_value": ["seed-original"]})
    output = attempt_input.copy()
    output["resampled_value"] = "fresh-attempt-value"
    output["accepted"] = True

    accepted, _ = classify_attempt(
        output=output,
        attempt_input=attempt_input,
        manifest=manifest,
        predicate_column="accepted",
    )

    assert accepted["resampled_value"].tolist() == ["fresh-attempt-value"]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lazy.pd.DataFrame({"accepted": [True]}), "missing internal slot column"),
        (lazy.pd.DataFrame({"slot": [0]}), "missing predicate column"),
        (
            lazy.pd.DataFrame({"slot": [0, 0], "accepted": [True, True]}),
            "duplicate rows for one or more slot IDs",
        ),
        (lazy.pd.DataFrame({"slot": [2], "accepted": [True]}), "unknown slot IDs"),
    ],
)
def test_classification_rejects_outputs_that_break_slot_identity(output: Any, message: str) -> None:
    manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=3,
        policy=RetryUntil(predicate_column="accepted", max_attempts=1).model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
    )
    with pytest.raises(DataDesignerWorkflowError, match=message):
        classify_attempt(
            output=output,
            attempt_input=lazy.pd.DataFrame({"slot": [0, 1]}),
            manifest=manifest,
            predicate_column="accepted",
        )


@pytest.mark.parametrize(
    ("values", "target", "message"),
    [
        ([True], 2, "non-integer slot ID"),
        ([1.0], 2, "non-integer slot ID"),
        ([-1], 2, "out-of-range slot ID"),
        ([2], 2, "out-of-range slot ID"),
    ],
)
def test_slot_ids_are_strict_bounded_integers(values: list[Any], target: int, message: str) -> None:
    with pytest.raises(DataDesignerWorkflowError, match=message):
        normalize_slot_ids(lazy.pd.Series(values), target, "test slots")


def test_immutable_attempt_input_rejects_changed_slots_or_schema(tmp_path: Path) -> None:
    path = tmp_path / "attempt" / "input.parquet"
    original = lazy.pd.DataFrame({"slot": [0], "value": ["first"]})
    write_or_validate_attempt_input(original, path, "slot")
    write_or_validate_attempt_input(original.copy(), path, "slot")

    with pytest.raises(DataDesignerWorkflowError, match="Stored immutable attempt input"):
        write_or_validate_attempt_input(lazy.pd.DataFrame({"slot": [1], "other": ["changed"]}), path, "slot")


def test_accepted_partition_validation_detects_missing_duplicates_counts_and_overlap(tmp_path: Path) -> None:
    with pytest.raises(DataDesignerWorkflowError, match="Accepted partition is missing"):
        read_accepted_slot_ids(tmp_path / "missing.parquet", "slot")

    duplicate_path = tmp_path / "duplicate.parquet"
    lazy.pd.DataFrame({"slot": [0, 0]}).to_parquet(duplicate_path, index=False)
    with pytest.raises(DataDesignerWorkflowError, match="contains duplicate slot IDs"):
        read_accepted_slot_ids(duplicate_path, "slot")

    count_path = tmp_path / "accepted" / "attempt-000.parquet"
    count_path.parent.mkdir()
    lazy.pd.DataFrame({"slot": [0]}).to_parquet(count_path, index=False)
    count_manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=1,
        policy=RetryUntil(predicate_column="accepted", max_attempts=2).model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
        attempts=[AttemptManifest.model_validate(_attempt_payload() | {"accepted_records": 0, "false_records": 1})],
    )
    with pytest.raises(DataDesignerWorkflowError, match="does not match its manifest record count"):
        load_committed_accepted_ids(tmp_path, count_manifest)

    second_path = tmp_path / "accepted" / "attempt-001.parquet"
    lazy.pd.DataFrame({"slot": [0]}).to_parquet(second_path, index=False)
    overlap_manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=2,
        policy=RetryUntil(predicate_column="accepted", max_attempts=2).model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
        attempts=[
            AttemptManifest.model_validate(
                _attempt_payload()
                | {
                    "input_records": 2,
                    "output_records": 2,
                    "accepted_records": 1,
                    "false_records": 1,
                }
            ),
            AttemptManifest.model_validate(_attempt_payload()),
        ],
    )
    with pytest.raises(DataDesignerWorkflowError, match="Slots were accepted more than once"):
        load_committed_accepted_ids(tmp_path, overlap_manifest)


def test_finalization_cleanup_removes_only_an_ambiguous_started_publication(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "metadata.json").write_text("{not-json", encoding="utf-8")
    clear_ambiguous_finalization(corrupt)
    assert (corrupt / "metadata.json").exists()

    started = tmp_path / "started"
    final_dataset = started / "parquet-files"
    final_dataset.mkdir(parents=True)
    (final_dataset / "batch_000000.parquet").write_bytes(b"partial")
    (started / "metadata.json").write_text(json.dumps({"post_generation_state": "started"}), encoding="utf-8")
    (started / "builder-config.json").write_text("partial", encoding="utf-8")

    clear_ambiguous_finalization(started)

    assert not final_dataset.exists()
    assert not (started / "metadata.json").exists()


def test_model_usage_is_aggregated_across_base_attempts_and_finalization() -> None:
    first_usage = {
        "token_usage": {"input_tokens": 2, "output_tokens": 3},
        "request_usage": {"successful_requests": 1, "failed_requests": 0},
        "tool_usage": {
            "total_tool_calls": 1,
            "total_tool_call_turns": 1,
            "total_generations": 1,
            "generations_with_tools": 1,
        },
        "image_usage": {"total_images": 1},
    }
    second_usage = {
        "token_usage": {"input_tokens": 5, "output_tokens": 7},
        "request_usage": {"successful_requests": 2, "failed_requests": 1},
        "tool_usage": {
            "total_tool_calls": 2,
            "total_tool_call_turns": 1,
            "total_generations": 2,
            "generations_with_tools": 1,
        },
        "image_usage": {"total_images": 3},
    }
    manifest = RetryManifest(
        fingerprint="fingerprint",
        target_records=1,
        policy=RetryUntil(predicate_column="accepted", max_attempts=1).model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
        base_model_usage={"model": first_usage},
        attempts=[AttemptManifest.model_validate(_attempt_payload() | {"model_usage": {"model": second_usage}})],
        final_model_usage={"model": first_usage},
    )

    aggregate = aggregate_model_usage(manifest)["model"]

    assert aggregate["token_usage"]["input_tokens"] == 9
    assert aggregate["token_usage"]["output_tokens"] == 13
    assert aggregate["request_usage"] == {
        "successful_requests": 4,
        "failed_requests": 1,
        "total_requests": 5,
    }
    assert aggregate["tool_usage"]["total_tool_calls"] == 4
    assert aggregate["image_usage"]["total_images"] == 5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (lazy.np.bool_(True), True),
        (lazy.np.bool_(False), False),
        (None, None),
        (lazy.pd.NA, None),
        (lazy.np.nan, None),
    ],
)
def test_strict_predicate_accepts_only_scalar_boolean_or_null(value: Any, expected: bool | None) -> None:
    assert strict_predicate_outcome(value) is expected


@pytest.mark.parametrize(
    "value",
    [
        1,
        0,
        "true",
        "false",
        [],
        {},
        lazy.np.array([True]),
    ],
)
def test_strict_predicate_rejects_truthy_coercions_and_non_scalars(value: Any) -> None:
    with pytest.raises(DataDesignerWorkflowError, match="strict scalar booleans or null"):
        strict_predicate_outcome(value)


def test_preserved_local_seed_images_are_copied_into_attempt_run(
    tmp_path: Path,
) -> None:
    seed_root = tmp_path / "seed"
    image_path = seed_root / "images" / "nested" / "image.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"seed-image")
    attempt_input = lazy.pd.DataFrame(
        {
            "image": [
                {
                    "primary": "images/nested/image.png",
                    "duplicates": lazy.np.array(["images/nested/image.png"]),
                }
            ]
        }
    )
    run_path = tmp_path / "stage" / ATTEMPTS_DIRECTORY / "attempt-000" / "run"

    copy_preserved_seed_media(
        attempt_input=attempt_input,
        seed_column_names=["image"],
        source_root=seed_root,
        run_path=run_path,
    )

    assert (run_path / "images" / "nested" / "image.png").read_bytes() == b"seed-image"


@pytest.mark.parametrize(
    ("case", "value", "message"),
    [
        ("unsafe", "images/../secret.png", "is unsafe"),
        ("unresolvable", "images/image.png", "cannot resolve preserved"),
        ("missing", "images/missing.png", "is missing beneath"),
    ],
)
def test_preserved_local_seed_media_rejects_unsafe_or_unresolvable_paths(
    case: str,
    value: str,
    message: str,
    tmp_path: Path,
) -> None:
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    if case == "missing":
        (seed_root / "images").mkdir()

    with pytest.raises(DataDesignerWorkflowError, match=message):
        copy_preserved_seed_media(
            attempt_input=lazy.pd.DataFrame({"image": [value]}),
            seed_column_names=["image"],
            source_root=None if case == "unresolvable" else seed_root,
            run_path=tmp_path / "attempt-run",
        )


def test_runner_uses_durable_base_media_after_original_seed_media_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_configs: list[ModelConfig],
    stub_model_providers: list[ModelProvider],
) -> None:
    seed_root = tmp_path / "seed"
    dataset_path = seed_root / "data.parquet"
    image_path = seed_root / "images" / "nested" / "image.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"seed-image")
    lazy.pd.DataFrame({"image": ["images/nested/image.png"]}).to_parquet(dataset_path, index=False)

    call_count = 0

    @custom_column_generator(required_columns=["image"], side_effect_columns=["accepted"])
    def generate(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        row["generated"] = f"attempt-{call_count}"
        row["accepted"] = call_count >= 2
        return row

    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.with_seed_dataset(
        LocalFileSeedSource(path=str(dataset_path)),
        sampling_strategy=SamplingStrategy.ORDERED,
    )
    builder.add_column(CustomColumnConfig(name="generated", generator_function=generate))

    artifact_path = tmp_path / "artifacts"
    data_designer = _real_data_designer(artifact_path, stub_model_providers)
    attempt_runner = RecordRetryAttemptRunner(data_designer)
    original_run_attempt = attempt_runner.run_attempt
    stage_path = artifact_path / "records"
    durable_media_path = stage_path / BASE_COHORT_PATH.parent / "images" / "nested" / "image.png"

    def remove_original_before_first_attempt(**kwargs: Any) -> AttemptManifest:
        if image_path.exists():
            assert durable_media_path.read_bytes() == b"seed-image"
            image_path.unlink()
        return original_run_attempt(**kwargs)

    monkeypatch.setattr(attempt_runner, "run_attempt", remove_original_before_first_attempt)

    result = RecordRetryRunner(data_designer, attempt_runner=attempt_runner).run(
        config_builder=builder,
        num_records=1,
        dataset_name="records",
        artifact_path=artifact_path,
        policy=RetryUntil(predicate_column="accepted", max_attempts=2),
        fingerprint="stable-fingerprint",
        resume=ResumeMode.NEVER,
        workflow_resume=ResumeMode.NEVER,
    )

    assert call_count == 2
    assert not image_path.exists()
    assert durable_media_path.read_bytes() == b"seed-image"
    assert (
        stage_path / ATTEMPTS_DIRECTORY / "attempt-000" / "run" / "images" / "nested" / "image.png"
    ).read_bytes() == b"seed-image"
    assert (
        stage_path / ATTEMPTS_DIRECTORY / "attempt-001" / "run" / "images" / "nested" / "image.png"
    ).read_bytes() == b"seed-image"
    assert result.load_dataset()["image"].tolist() == ["images/attempt-001/nested/image.png"]


def test_media_packaging_copies_and_rewrites_nested_attempt_media_without_escaping_root(tmp_path: Path) -> None:
    attempt_root = tmp_path / "attempt-run"
    stage_path = tmp_path / "stage"
    source = attempt_root / "images" / "nested" / "image.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")
    outside = attempt_root / "outside.png"
    outside.write_bytes(b"outside-bytes")
    accepted = lazy.pd.DataFrame(
        {
            "plain": ["images/nested/image.png"],
            "nested": [
                {
                    "list": ["images/nested/image.png", "images/missing.png"],
                    "tuple": ("images/nested/image.png",),
                    "array": lazy.np.array(["images/nested/image.png"]),
                    "traversal": "images/../outside.png",
                }
            ],
        }
    )

    packaged = package_accepted_media(
        accepted,
        attempt_root=attempt_root,
        stage_path=stage_path,
        attempt_name="attempt-007",
    )

    rewritten = "images/attempt-007/nested/image.png"
    assert packaged.loc[0, "plain"] == rewritten
    nested = packaged.loc[0, "nested"]
    assert nested["list"] == [rewritten, "images/missing.png"]
    assert nested["tuple"] == (rewritten,)
    assert nested["array"] == [rewritten]
    assert nested["traversal"] == "images/../outside.png"
    assert (stage_path / rewritten).read_bytes() == b"image-bytes"
    assert not (stage_path / "images" / "attempt-007" / "outside.png").exists()


def test_empty_media_partition_is_returned_without_creating_media_directory(tmp_path: Path) -> None:
    accepted = lazy.pd.DataFrame({"image": lazy.pd.Series(dtype="object")})

    packaged = package_accepted_media(
        accepted,
        attempt_root=tmp_path / "attempt",
        stage_path=tmp_path / "stage",
        attempt_name="attempt-000",
    )

    assert packaged is accepted
    assert not (tmp_path / "stage").exists()
