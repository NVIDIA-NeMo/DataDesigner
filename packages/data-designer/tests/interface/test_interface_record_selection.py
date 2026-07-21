# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig, ExpressionColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.record_selection import RecordSelectionConfig, RecordSelectionExhaustion
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.column_generators.generators.expression import ExpressionColumnGenerator
from data_designer.engine.column_generators.generators.samplers import SamplerColumnGenerator
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.dataset_builders.errors import DatasetGenerationError, RecordSelectionEarlyShutdownError
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import (
    DataDesignerEarlyShutdownError,
    DataDesignerGenerationError,
    DataDesignerRecordSelectionExhaustedError,
)


@custom_column_generator(required_columns=["value"])
def _raise_predicate_generation_error(row: dict) -> dict:
    raise ValueError("deterministic predicate failure")


@custom_column_generator(required_columns=["value"])
def _select_good_value_or_raise(row: dict) -> dict:
    if row["value"] == "bad":
        raise ValueError("bad candidate")
    return {**row, "keep": row["value"] == "good"}


def _designer(tmp_path: Path) -> DataDesigner:
    managed_assets = tmp_path / "managed-assets"
    managed_assets.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path,
        managed_assets_path=managed_assets,
        auto_configure_logging=False,
    )
    designer.set_run_config(RunConfig(buffer_size=2, display_tui=False))
    return designer


def _builder(*, predicate: str, cap: int, on_exhausted: RecordSelectionExhaustion) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="keep",
            expr=predicate,
            dtype="bool",
            drop=True,
        )
    )
    return builder.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=cap,
            on_exhausted=on_exhausted,
        )
    )


def test_create_selects_exact_target_across_batches_and_trims(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    results = designer.create(
        _builder(predicate="{{ true }}", cap=10, on_exhausted=RecordSelectionExhaustion.RAISE),
        num_records=3,
    )

    assert results.count_records() == 3
    assert results.load_dataset().columns.tolist() == ["value"]
    metadata = results.artifact_storage.read_metadata()
    assert metadata["actual_num_records"] == 3
    assert metadata["post_generation_state"] == "complete"
    assert metadata["record_selection"]["candidate_records_generated"] == 4
    assert metadata["record_selection"]["candidate_batches_completed"] == 2
    assert metadata["record_selection"]["trimmed_accepted_records"] == 1
    assert metadata["record_selection"]["selection_satisfied"] is True
    markers = sorted(results.artifact_storage.selection_checkpoints_path.glob("batch_*.json"))
    assert len(markers) == 2
    assert sum(json.loads(path.read_text())["candidate_records"] for path in markers) == 4


def test_generators_log_pre_generation_once_across_candidate_batches(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    with (
        patch.object(SamplerColumnGenerator, "log_pre_generation") as sampler_log,
        patch.object(ExpressionColumnGenerator, "log_pre_generation") as expression_log,
    ):
        results = designer.create(
            _builder(predicate="{{ true }}", cap=10, on_exhausted=RecordSelectionExhaustion.RAISE),
            num_records=3,
        )

    assert results.artifact_storage.read_metadata()["record_selection"]["candidate_batches_completed"] == 2
    sampler_log.assert_called_once_with()
    expression_log.assert_called_once_with()


def test_ordered_seed_offsets_advance_in_candidate_coordinate_space(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = DataDesignerConfigBuilder().with_seed_dataset(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"ordinal": list(range(8))}))
    )
    builder.add_column(ExpressionColumnConfig(name="ordinal_copy", expr="{{ ordinal }}", dtype="int"))
    builder.add_column(
        ExpressionColumnConfig(
            name="keep",
            expr="{{ ordinal_copy % 2 == 1 }}",
            dtype="bool",
            drop=True,
        )
    )
    builder.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=8))

    results = designer.create(builder, num_records=3)

    assert results.load_dataset()["ordinal_copy"].tolist() == [1, 3, 5]
    metadata = results.artifact_storage.read_metadata()
    assert metadata["record_selection"]["candidate_records_generated"] == 6


def test_create_returns_schema_bearing_empty_partial_and_skips_profiling(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = _builder(
        predicate="{{ false }}",
        cap=5,
        on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
    )
    results = designer.create(
        builder,
        num_records=2,
        dataset_name="empty-partial",
    )

    dataset = results.load_dataset()
    assert dataset.empty
    assert dataset.columns.tolist() == ["value"]
    assert results.count_records() == 0
    assert results.load_analysis() is None
    metadata = results.artifact_storage.read_metadata()
    assert metadata["column_statistics"] == []
    assert metadata["record_selection"]["candidate_records_generated"] == 5
    assert metadata["record_selection"]["candidate_batches_completed"] == 3
    assert metadata["record_selection"]["selection_exhausted"] is True
    assert len(list(results.artifact_storage.selection_checkpoints_path.glob("batch_*.json"))) == 3

    for path in results.artifact_storage.final_dataset_path.glob("*.parquet"):
        path.unlink()
    rebuilt = designer.create(
        builder,
        num_records=2,
        dataset_name="empty-partial",
        resume=ResumeMode.ALWAYS,
    )
    assert rebuilt.load_dataset().empty
    assert rebuilt.load_dataset().columns.tolist() == ["value"]


def test_create_profiles_non_empty_partial_against_requested_target(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = DataDesignerConfigBuilder().with_seed_dataset(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"ordinal": [0, 1, 2]}))
    )
    builder.add_column(ExpressionColumnConfig(name="ordinal_copy", expr="{{ ordinal }}", dtype="int"))
    builder.add_column(
        ExpressionColumnConfig(
            name="keep",
            expr="{{ ordinal_copy == 0 }}",
            dtype="bool",
            drop=True,
        )
    )
    builder.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=3,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        )
    )

    results = designer.create(builder, num_records=2)

    analysis = results.load_analysis()
    assert analysis is not None
    assert analysis.num_records == 1
    assert analysis.target_num_records == 2
    assert analysis.percent_complete == pytest.approx(50.0)


def test_create_raises_structured_exhaustion_error(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    with pytest.raises(DataDesignerRecordSelectionExhaustedError) as exc_info:
        designer.create(
            _builder(predicate="{{ false }}", cap=5, on_exhausted=RecordSelectionExhaustion.RAISE),
            num_records=2,
        )
    error = exc_info.value
    assert error.target_records == 2
    assert error.accepted_records == 0
    assert error.candidate_records == 5
    assert error.max_candidate_records == 5


def test_selection_early_shutdown_remains_a_typed_public_error(tmp_path: Path) -> None:
    designer = _designer(tmp_path)

    def stop_after_checkpoint(_builder: DatasetBuilder, **_kwargs: object) -> None:
        raise RecordSelectionEarlyShutdownError()

    with (
        patch.object(DatasetBuilder, "build", autospec=True, side_effect=stop_after_checkpoint),
        pytest.raises(DataDesignerEarlyShutdownError, match="Record selection stopped"),
    ):
        designer.create(
            _builder(predicate="{{ true }}", cap=3, on_exhausted=RecordSelectionExhaustion.RAISE),
            num_records=2,
        )


def test_selection_prior_early_shutdown_does_not_mask_later_generation_error(tmp_path: Path) -> None:
    designer = _designer(tmp_path)

    def fail_after_prior_early_shutdown(builder: DatasetBuilder, **_kwargs: object) -> None:
        builder._early_shutdown = True
        raise DatasetGenerationError("publication failed")

    with (
        patch.object(DatasetBuilder, "build", autospec=True, side_effect=fail_after_prior_early_shutdown),
        pytest.raises(DataDesignerGenerationError, match="publication failed") as exc_info,
    ):
        designer.create(
            _builder(predicate="{{ true }}", cap=3, on_exhausted=RecordSelectionExhaustion.RAISE),
            num_records=2,
        )

    assert not isinstance(exc_info.value, DataDesignerEarlyShutdownError)


def test_selection_nonretryable_empty_partial_remains_failure_across_fresh_resume(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    run_config = RunConfig(buffer_size=1, disable_early_shutdown=True, display_tui=False)
    designer.set_run_config(run_config)
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    builder.add_column(
        CustomColumnConfig(
            name="keep",
            generator_function=_raise_predicate_generation_error,
            drop=True,
        )
    )
    builder.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=1,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        )
    )

    with pytest.raises(DataDesignerGenerationError, match="deterministic predicate failure"):
        designer.create(builder, num_records=1, dataset_name="nonretryable-empty-selection")

    marker_path = tmp_path / "nonretryable-empty-selection" / "selection-checkpoints" / "batch_00000.json"
    marker = json.loads(marker_path.read_text())
    assert marker["non_retryable_error"] == (
        "CustomColumnGenerationError: "
        "Custom generator function failed for column 'keep': deterministic predicate failure"
    )

    resumed_designer = DataDesigner(
        artifact_path=tmp_path,
        managed_assets_path=tmp_path / "managed-assets",
        auto_configure_logging=False,
    )
    resumed_designer.set_run_config(run_config)
    with pytest.raises(DataDesignerGenerationError, match="deterministic predicate failure"):
        resumed_designer.create(
            builder,
            num_records=1,
            dataset_name="nonretryable-empty-selection",
            resume=ResumeMode.ALWAYS,
        )


def test_selection_nonretryable_failure_does_not_override_structured_exhaustion(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    designer.set_run_config(RunConfig(buffer_size=1, disable_early_shutdown=True, display_tui=False))
    builder = DataDesignerConfigBuilder().with_seed_dataset(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"value": ["bad"]}))
    )
    builder.add_column(ExpressionColumnConfig(name="value_copy", expr="{{ value }}", dtype="str"))
    builder.add_column(CustomColumnConfig(name="keep", generator_function=_select_good_value_or_raise, drop=True))
    builder.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=1,
            on_exhausted=RecordSelectionExhaustion.RAISE,
        )
    )

    with pytest.raises(DataDesignerRecordSelectionExhaustedError):
        designer.create(builder, num_records=1)


def test_selection_nonretryable_failure_allows_nonempty_partial(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    designer.set_run_config(RunConfig(buffer_size=2, disable_early_shutdown=True, display_tui=False))
    builder = DataDesignerConfigBuilder().with_seed_dataset(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"value": ["bad", "good"]}))
    )
    builder.add_column(ExpressionColumnConfig(name="value_copy", expr="{{ value }}", dtype="str"))
    builder.add_column(CustomColumnConfig(name="keep", generator_function=_select_good_value_or_raise, drop=True))
    builder.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=2,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        )
    )

    results = designer.create(builder, num_records=2)

    assert results.load_dataset()["value_copy"].tolist() == ["good"]
    assert results.artifact_storage.read_metadata()["record_selection"]["accepted_records"] == 1


def test_preview_rejects_record_selection_before_generation(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    with pytest.raises(DataDesignerGenerationError, match="preview.*does not support record selection"):
        designer.preview(
            _builder(predicate="{{ true }}", cap=3, on_exhausted=RecordSelectionExhaustion.RAISE),
            num_records=2,
        )


def test_create_rejects_candidate_cap_below_target(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    with pytest.raises(DataDesignerGenerationError, match="greater than or equal"):
        designer.create(
            _builder(predicate="{{ true }}", cap=2, on_exhausted=RecordSelectionExhaustion.RAISE),
            num_records=3,
        )


def test_completed_selection_resumes_only_with_same_target(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = _builder(predicate="{{ true }}", cap=5, on_exhausted=RecordSelectionExhaustion.RAISE)
    first = designer.create(builder, num_records=2, dataset_name="resume-selection")
    marker_mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in first.artifact_storage.selection_checkpoints_path.glob("batch_*.json")
    }

    resumed = designer.create(
        builder,
        num_records=2,
        dataset_name="resume-selection",
        resume=ResumeMode.ALWAYS,
    )
    assert resumed.count_records() == 2
    assert marker_mtimes == {
        path.name: path.stat().st_mtime_ns
        for path in resumed.artifact_storage.selection_checkpoints_path.glob("batch_*.json")
    }

    with pytest.raises(DataDesignerGenerationError, match="must exactly match"):
        designer.create(
            builder,
            num_records=3,
            dataset_name="resume-selection",
            resume=ResumeMode.ALWAYS,
        )


def test_resume_continues_after_callback_crash_without_regenerating_committed_batch(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = _builder(predicate="{{ true }}", cap=8, on_exhausted=RecordSelectionExhaustion.RAISE)

    def crash_after_commit(_path: Path) -> None:
        raise RuntimeError("simulated callback crash")

    with pytest.raises(DataDesignerGenerationError, match="simulated callback crash"):
        designer.create(
            builder,
            num_records=3,
            dataset_name="interrupted-selection",
            on_batch_complete=crash_after_commit,
        )

    checkpoints = tmp_path / "interrupted-selection" / "selection-checkpoints"
    first_marker = checkpoints / "batch_00000.json"
    first_marker_mtime = first_marker.stat().st_mtime_ns

    resumed = designer.create(
        builder,
        num_records=3,
        dataset_name="interrupted-selection",
        resume=ResumeMode.ALWAYS,
    )

    assert resumed.count_records() == 3
    assert first_marker.stat().st_mtime_ns == first_marker_mtime
    assert len(list(checkpoints.glob("batch_*.json"))) == 2
    metadata = resumed.artifact_storage.read_metadata()
    assert metadata["record_selection"]["candidate_records_generated"] == 4


def test_resume_rebuilds_incomplete_publication_from_immutable_partitions(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = _builder(predicate="{{ true }}", cap=4, on_exhausted=RecordSelectionExhaustion.RAISE)
    first = designer.create(builder, num_records=2, dataset_name="publication-recovery")
    metadata = first.artifact_storage.read_metadata()
    metadata["post_generation_state"] = "started"
    metadata["post_generation_processed"] = False
    first.artifact_storage.write_metadata(metadata)
    for path in first.artifact_storage.final_dataset_path.glob("*.parquet"):
        path.unlink()

    resumed = designer.create(
        builder,
        num_records=2,
        dataset_name="publication-recovery",
        resume=ResumeMode.ALWAYS,
    )

    assert resumed.count_records() == 2
    assert resumed.artifact_storage.read_metadata()["post_generation_state"] == "complete"

    for path in resumed.artifact_storage.final_dataset_path.glob("*.parquet"):
        path.unlink()
    rebuilt_from_complete = designer.create(
        builder,
        num_records=2,
        dataset_name="publication-recovery",
        resume=ResumeMode.ALWAYS,
    )
    assert rebuilt_from_complete.count_records() == 2


def test_if_possible_runtime_mismatch_clears_selection_artifacts_and_restarts(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = _builder(predicate="{{ true }}", cap=8, on_exhausted=RecordSelectionExhaustion.RAISE)
    first = designer.create(builder, num_records=2, dataset_name="if-possible-selection")
    old_partition = first.artifact_storage.selection_partition_path(0)
    old_partition.write_bytes(b"stale selection partition")

    restarted = designer.create(
        builder,
        num_records=3,
        dataset_name="if-possible-selection",
        resume=ResumeMode.IF_POSSIBLE,
    )

    assert restarted.artifact_storage.base_dataset_path == first.artifact_storage.base_dataset_path
    assert restarted.count_records() == 3
    assert lazy.pq.read_metadata(restarted.artifact_storage.selection_partition_path(0)).num_rows == 2
    metadata = restarted.artifact_storage.read_metadata()
    assert metadata["target_num_records"] == 3
    assert metadata["record_selection"]["candidate_records_generated"] == 4


def test_resume_regenerates_work_without_a_durable_checkpoint(tmp_path: Path) -> None:
    designer = _designer(tmp_path)
    builder = _builder(predicate="{{ true }}", cap=4, on_exhausted=RecordSelectionExhaustion.RAISE)
    first = designer.create(builder, num_records=2, dataset_name="missing-checkpoint")
    first.artifact_storage.selection_checkpoint_path(0).unlink()

    resumed = designer.create(
        builder,
        num_records=2,
        dataset_name="missing-checkpoint",
        resume=ResumeMode.ALWAYS,
    )

    assert resumed.count_records() == 2
    assert resumed.artifact_storage.selection_checkpoint_path(0).is_file()
