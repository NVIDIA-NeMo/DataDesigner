# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from fnmatch import fnmatch
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.processors import ProcessorType
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.models.usage import ModelUsageStats
from data_designer.engine.storage.artifact_storage import (
    FINAL_DATASET_FOLDER_NAME,
    METADATA_FILENAME,
    PROCESSORS_OUTPUTS_FOLDER_NAME,
    SDG_CONFIG_FILENAME,
    ArtifactStorage,
    ResumeMode,
)
from data_designer.interface.cohort_retry import RetryUntil, SamplerRetryMode
from data_designer.interface.cohort_retry_builders import CohortRetryBuilderProjection
from data_designer.interface.cohort_retry_state import (
    BASE_COHORT_PATH,
    COALESCED_ACCEPTED_PATH,
    FINAL_COMPLETION_FILENAME,
    AttemptCompletion,
    RetryManifest,
    copy_file_atomic,
    get_attempt_accepted_path,
    read_final_completion,
    read_retry_manifest,
    write_parquet_atomic,
)
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.results import _load_analysis_from_artifact_storage

if TYPE_CHECKING:
    import pandas as pd

_DROPPED_COLUMNS_FOLDER_NAME = "dropped-columns-parquet-files"
_PARTIAL_RESULTS_FOLDER_NAME = "tmp-partial-parquet-files"


def is_completed_retry_state_reusable(
    *,
    stage_path: Path,
    fingerprint: str,
    policy: RetryUntil,
    target_records: int,
) -> bool:
    """Return whether a completed workflow stage has a coherent retry terminal state."""
    try:
        manifest = read_retry_manifest(stage_path)
        if (
            manifest.status != "complete"
            or manifest.fingerprint != fingerprint
            or manifest.policy != policy.to_dict()
            or manifest.target_records != target_records
        ):
            return False

        storage = ArtifactStorage(
            artifact_path=stage_path.parent,
            dataset_name=stage_path.name,
            resume=ResumeMode.ALWAYS,
        )
        batch_files = list(storage.final_dataset_path.glob("batch_*.parquet"))
        if not batch_files or count_storage_records(storage) != manifest.accepted_records:
            return False

        metadata = storage.read_metadata()
        if not isinstance(metadata, dict):
            return False
        expected_usage = aggregate_model_usage(manifest)
        if (
            type(metadata.get("target_num_records")) is not int
            or metadata["target_num_records"] != manifest.target_records
            or type(metadata.get("original_target_num_records")) is not int
            or metadata["original_target_num_records"] != manifest.target_records
            or type(metadata.get("actual_num_records")) is not int
            or metadata["actual_num_records"] != manifest.accepted_records
            or metadata.get("cohort_retry") != metadata_retry_summary(manifest, expected_usage)
        ):
            return False

        config_payload = json.loads((stage_path / SDG_CONFIG_FILENAME).read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            return False

        completion_path = stage_path / FINAL_COMPLETION_FILENAME
        if manifest.accepted_records == 0:
            return not completion_path.exists() and _load_analysis_from_artifact_storage(storage) is None

        completion = read_final_completion(completion_path)
        analysis = _load_analysis_from_artifact_storage(storage)
        return bool(
            completion is not None
            and completion.accepted_records == manifest.accepted_records
            and completion.model_usage == manifest.final_model_usage
            and analysis is not None
            and analysis.num_records == manifest.accepted_records
            and analysis.target_num_records == manifest.target_records
        )
    except (
        DataDesignerWorkflowError,
        FileNotFoundError,
        UnicodeError,
        json.JSONDecodeError,
        OSError,
        ValidationError,
        lazy.pa.ArrowException,
    ):
        return False


def load_and_validate_base_cohort(stage_path: Path, manifest: RetryManifest) -> pd.DataFrame:
    """Load a persisted base cohort and validate its stable slot identity."""
    path = stage_path / BASE_COHORT_PATH
    if not path.is_file():
        raise DataDesignerWorkflowError(f"Cannot resume cohort retry: base cohort is missing at {str(path)!r}.")
    try:
        base_df = lazy.pd.read_parquet(path)
    except Exception as exc:
        raise DataDesignerWorkflowError(f"Cannot read base cohort at {str(path)!r}: {exc}") from exc
    if len(base_df) != manifest.target_records or manifest.slot_column not in base_df:
        raise DataDesignerWorkflowError("Cannot resume cohort retry: base cohort shape is incompatible.")
    slot_ids = normalize_slot_ids(base_df[manifest.slot_column], manifest.target_records, "base cohort")
    if slot_ids != list(range(manifest.target_records)):
        raise DataDesignerWorkflowError("Cannot resume cohort retry: base cohort slot IDs are not contiguous.")
    return base_df


def classify_attempt(
    *,
    output: pd.DataFrame,
    attempt_input: pd.DataFrame,
    manifest: RetryManifest,
    predicate_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify one attempt and return accepted rows with durable accounting counts."""
    slot_column = manifest.slot_column
    if len(output) and slot_column not in output:
        raise DataDesignerWorkflowError(f"Attempt output is missing internal slot column {slot_column!r}.")
    if len(output) and predicate_column not in output:
        raise DataDesignerWorkflowError(f"Attempt output is missing predicate column {predicate_column!r}.")

    expected_ids = set(normalize_slot_ids(attempt_input[slot_column], manifest.target_records, "attempt input"))
    if len(output):
        output_ids = normalize_slot_ids(output[slot_column], manifest.target_records, "attempt output")
        if len(output_ids) != len(set(output_ids)):
            raise DataDesignerWorkflowError("Attempt output contains duplicate rows for one or more slot IDs.")
        unknown = set(output_ids).difference(expected_ids)
        if unknown:
            raise DataDesignerWorkflowError(f"Attempt output contains unknown slot IDs: {sorted(unknown)!r}.")
        validate_stable_attempt_values(
            output=output,
            attempt_input=attempt_input,
            slot_column=slot_column,
        )
    else:
        output_ids = []

    accepted_mask: list[bool] = []
    false_records = 0
    null_records = 0
    for value in output[predicate_column].tolist() if len(output) else []:
        outcome = strict_predicate_outcome(value)
        accepted_mask.append(outcome is True)
        if outcome is False:
            false_records += 1
        elif outcome is None:
            null_records += 1

    accepted = output.loc[accepted_mask].copy() if len(output) else output.copy()
    return accepted, {
        "output_records": len(output),
        "accepted_records": len(accepted),
        "false_records": false_records,
        "null_records": null_records,
        "missing_records": len(expected_ids.difference(output_ids)),
    }


def retry_bounds_exhausted(
    policy: RetryUntil,
    manifest: RetryManifest,
    pending_records: int,
) -> bool:
    """Return whether another complete pending-cohort attempt would exceed a retry bound."""
    if policy.max_attempts is not None and len(manifest.attempts) >= policy.max_attempts:
        return True
    return (
        policy.max_candidate_records is not None
        and manifest.candidate_records + pending_records > policy.max_candidate_records
    )


def load_committed_accepted_ids(stage_path: Path, manifest: RetryManifest) -> set[int]:
    """Load and validate the non-overlapping slot IDs committed by the manifest."""
    accepted_ids: set[int] = set()
    for attempt_index, attempt in enumerate(manifest.attempts):
        path = stage_path / get_attempt_accepted_path(attempt_index)
        ids = read_accepted_slot_ids(path, manifest.slot_column, manifest.target_records)
        if len(ids) != attempt.accepted_records:
            raise DataDesignerWorkflowError(
                f"Accepted partition {str(path)!r} does not match its manifest record count."
            )
        overlap = accepted_ids.intersection(ids)
        if overlap:
            raise DataDesignerWorkflowError(f"Slots were accepted more than once: {sorted(overlap)!r}.")
        accepted_ids.update(ids)
    return accepted_ids


def read_accepted_slot_ids(path: Path, slot_column: str, target_records: int | None = None) -> set[int]:
    """Read one accepted partition's unique, bounded slot IDs."""
    if not path.is_file():
        raise DataDesignerWorkflowError(f"Accepted partition is missing at {str(path)!r}.")
    df = lazy.pd.read_parquet(path, columns=[slot_column])
    ids = normalize_slot_ids(df[slot_column], target_records, "accepted partition")
    if len(ids) != len(set(ids)):
        raise DataDesignerWorkflowError(f"Accepted partition {str(path)!r} contains duplicate slot IDs.")
    return set(ids)


def coalesce_accepted(
    stage_path: Path,
    manifest: RetryManifest,
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    """Coalesce committed accepted partitions in stable original-cohort order."""
    partitions = [
        lazy.pd.read_parquet(stage_path / get_attempt_accepted_path(attempt_index))
        for attempt_index in range(len(manifest.attempts))
    ]
    non_empty = [partition for partition in partitions if len(partition)]
    if non_empty:
        coalesced = lazy.pd.concat(non_empty, ignore_index=True)
        coalesced = coalesced.sort_values(manifest.slot_column, kind="stable").reset_index(drop=True)
    else:
        coalesced = partitions[0].head(0).copy() if partitions else base_df.head(0).copy()
        if manifest.attempt_column not in coalesced:
            coalesced[manifest.attempt_column] = lazy.pd.Series(dtype="int64")

    ids = normalize_slot_ids(coalesced[manifest.slot_column], manifest.target_records, "coalesced output")
    if len(ids) != len(set(ids)) or len(ids) != manifest.accepted_records:
        raise DataDesignerWorkflowError("Coalesced accepted output violates the one-row-per-slot invariant.")
    write_parquet_atomic(coalesced, stage_path / COALESCED_ACCEPTED_PATH)
    return coalesced


def normalize_slot_ids(series: pd.Series, target_records: int | None, label: str) -> list[int]:
    """Normalize strict integer slot IDs and enforce their optional target range."""
    normalized: list[int] = []
    for value in series.tolist():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise DataDesignerWorkflowError(f"{label} contains a non-integer slot ID: {value!r}.")
        slot_id = int(value)
        if slot_id < 0 or (target_records is not None and slot_id >= target_records):
            raise DataDesignerWorkflowError(f"{label} contains out-of-range slot ID {slot_id}.")
        normalized.append(slot_id)
    return normalized


def validate_stable_attempt_values(
    *,
    output: pd.DataFrame,
    attempt_input: pd.DataFrame,
    slot_column: str,
) -> None:
    """Require an attempt to preserve every seed/cohort value keyed by stable slot ID."""
    stable_columns = [column for column in attempt_input.columns if column != slot_column]
    missing_columns = [column for column in stable_columns if column not in output]
    if missing_columns:
        raise DataDesignerWorkflowError(f"Attempt output removed stable seed/cohort columns: {missing_columns!r}.")

    expected_by_slot = attempt_input.set_index(slot_column, drop=False)
    for _, row in output.iterrows():
        slot_id = int(row[slot_column])
        expected = expected_by_slot.loc[slot_id]
        for column in stable_columns:
            if not values_equal(row[column], expected[column]):
                raise DataDesignerWorkflowError(
                    f"Attempt output mutated stable seed/cohort column {column!r} for slot {slot_id}."
                )


def values_equal(left: Any, right: Any) -> bool:
    """Compare persisted scalar or nested dataframe cell values without truth coercion."""
    left_missing = is_scalar_missing(left)
    right_missing = is_scalar_missing(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(values_equal(left[key], right[key]) for key in left)
    sequence_types = (list, tuple, lazy.np.ndarray)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if not isinstance(left, sequence_types) or not isinstance(right, sequence_types):
            return False
        left_items = list(left)
        right_items = list(right)
        return len(left_items) == len(right_items) and all(
            values_equal(left_item, right_item) for left_item, right_item in zip(left_items, right_items)
        )
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(equal, (bool, lazy.np.bool_)) and bool(equal)


def is_scalar_missing(value: Any) -> bool:
    """Return whether a scalar dataframe cell uses a supported missing-value sentinel."""
    if value is None or value is lazy.pd.NA:
        return True
    try:
        missing = lazy.pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, lazy.np.bool_)) and bool(missing)


def strict_predicate_outcome(value: Any) -> bool | None:
    """Interpret only scalar Boolean or null predicate values."""
    if isinstance(value, (bool, lazy.np.bool_)):
        return bool(value)
    if value is None or value is lazy.pd.NA:
        return None
    try:
        missing = lazy.pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, lazy.np.bool_)) and bool(missing):
        return None
    raise DataDesignerWorkflowError(
        "retry_until predicate values must be strict scalar booleans or null; "
        f"received {value!r} ({type(value).__name__})."
    )


def empty_attempt_output(
    attempt_input: pd.DataFrame,
    projection: CohortRetryBuilderProjection | None,
) -> pd.DataFrame:
    """Create a schema-bearing empty attempt output after successful zero-row generation."""
    output = attempt_input.head(0).copy()
    if projection is not None:
        for column in projection.original_config.columns:
            for name in (column.name, *column.side_effect_columns):
                if name not in output:
                    output[name] = lazy.pd.Series(dtype="object")
    return output


def package_accepted_media(
    accepted: pd.DataFrame,
    *,
    attempt_root: Path,
    stage_path: Path,
    attempt_name: str,
) -> pd.DataFrame:
    """Copy accepted attempt media into the stage root and rewrite relative references."""
    if accepted.empty:
        return accepted
    packaged = accepted.copy()
    for column in packaged.columns:
        packaged[column] = packaged[column].map(
            lambda value: package_media_value(
                value,
                attempt_root=attempt_root,
                stage_path=stage_path,
                attempt_name=attempt_name,
            )
        )
    return packaged


def copy_preserved_seed_media(
    *,
    attempt_input: pd.DataFrame,
    seed_column_names: list[str],
    source_root: Path | None,
    run_path: Path,
) -> None:
    """Copy relative media referenced by preserved seed values into an internal run."""
    media_paths: set[Path] = set()
    for column in seed_column_names:
        if column not in attempt_input:
            continue
        for value in attempt_input[column].tolist():
            media_paths.update(relative_media_paths(value))
    if not media_paths:
        return

    resolved_run = run_path.resolve()
    missing_destinations: list[tuple[Path, Path]] = []
    for relative in sorted(media_paths):
        destination = (run_path / relative).resolve()
        if resolved_run not in destination.parents:
            raise DataDesignerWorkflowError(f"Preserved seed media destination {str(relative)!r} is unsafe.")
        if not destination.is_file():
            missing_destinations.append((relative, destination))
    if not missing_destinations:
        return

    if source_root is None:
        raise DataDesignerWorkflowError(
            "Cohort retry cannot resolve preserved 'images/...' seed media. Use a LocalFileSeedSource whose "
            "dataset has an adjacent images directory."
        )
    resolved_root = source_root.resolve()
    for relative, destination in missing_destinations:
        source = (source_root / relative).resolve()
        if resolved_root not in source.parents or not source.is_file():
            raise DataDesignerWorkflowError(
                f"Preserved seed media {str(relative)!r} is missing beneath {str(source_root)!r}."
            )
        copy_file_atomic(source, destination)


def relative_media_paths(value: Any) -> set[Path]:
    """Collect safe stage-relative ``images/...`` references from a nested value."""
    if isinstance(value, str) and value.startswith("images/"):
        relative = Path(value)
        if relative.is_absolute() or relative.parts[:1] != ("images",) or ".." in relative.parts:
            raise DataDesignerWorkflowError(f"Preserved seed media path {value!r} is unsafe.")
        return {relative}
    if isinstance(value, dict):
        paths: set[Path] = set()
        for item in value.values():
            paths.update(relative_media_paths(item))
        return paths
    if isinstance(value, lazy.np.ndarray):
        return relative_media_paths(value.tolist())
    if isinstance(value, (list, tuple)):
        paths = set()
        for item in value:
            paths.update(relative_media_paths(item))
        return paths
    return set()


def local_seed_media_root(config: DataDesignerConfig) -> Path | None:
    """Resolve the nearest local seed ancestor containing an adjacent images directory."""
    if config.seed_config is None or not isinstance(config.seed_config.source, LocalFileSeedSource):
        return None
    runtime_path = Path(config.seed_config.source.runtime_path)
    start = runtime_path if runtime_path.is_dir() else runtime_path.parent
    for candidate in (start, *start.parents):
        if (candidate / "images").is_dir():
            return candidate
    return None


def package_media_value(value: Any, *, attempt_root: Path, stage_path: Path, attempt_name: str) -> Any:
    """Copy and rewrite safe nested media references that resolve inside an attempt artifact."""
    if isinstance(value, str) and value.startswith("images/"):
        relative = Path(value)
        if relative.is_absolute() or relative.parts[:1] != ("images",) or ".." in relative.parts:
            return value
        source = (attempt_root / relative).resolve()
        resolved_root = attempt_root.resolve()
        if source.is_file() and resolved_root in source.parents:
            destination_relative = Path("images") / attempt_name / Path(*relative.parts[1:])
            destination = stage_path / destination_relative
            resolved_stage = stage_path.resolve()
            if resolved_stage not in destination.resolve().parents:
                return value
            copy_file_atomic(source, destination)
            return str(destination_relative)
        return value
    if isinstance(value, list):
        return [
            package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for item in value
        )
    if isinstance(value, dict):
        return {
            key: package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for key, item in value.items()
        }
    if isinstance(value, lazy.np.ndarray):
        return [
            package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for item in value.tolist()
        ]
    return value


def write_or_validate_attempt_input(df: pd.DataFrame, path: Path, slot_column: str) -> None:
    """Atomically persist an immutable attempt input, or validate its existing contents."""
    if not path.exists():
        write_parquet_atomic(df, path)
        return
    stored = lazy.pd.read_parquet(path)
    expected_ids = normalize_slot_ids(df[slot_column], None, "attempt input")
    stored_ids = normalize_slot_ids(stored[slot_column], None, "stored attempt input")
    if expected_ids != stored_ids or list(stored.columns) != list(df.columns):
        raise DataDesignerWorkflowError(f"Stored immutable attempt input at {str(path)!r} is incompatible.")
    for expected_row, stored_row in zip(
        df.itertuples(index=False, name=None), stored.itertuples(index=False, name=None)
    ):
        if len(expected_row) != len(stored_row) or any(
            not values_equal(expected, persisted) for expected, persisted in zip(expected_row, stored_row)
        ):
            raise DataDesignerWorkflowError(f"Stored immutable attempt input at {str(path)!r} was modified.")


def load_completed_attempt_output(
    *,
    run_path: Path,
    completion: AttemptCompletion,
    attempt_input: pd.DataFrame,
    projection: CohortRetryBuilderProjection,
) -> pd.DataFrame:
    """Load and validate an attempt output after its durable completion marker exists."""
    if completion.output_records == 0:
        return empty_attempt_output(attempt_input, projection)
    try:
        storage = ArtifactStorage(
            artifact_path=run_path.parent,
            dataset_name=run_path.name,
            resume=ResumeMode.ALWAYS,
        )
        output = storage.load_dataset()
    except Exception as exc:
        raise DataDesignerWorkflowError(f"Completed attempt output at {str(run_path)!r} cannot be read: {exc}") from exc
    if len(output) != completion.output_records:
        raise DataDesignerWorkflowError(
            f"Completed attempt output at {str(run_path)!r} contains {len(output)} records; "
            f"its completion marker records {completion.output_records}."
        )
    return output


def empty_publication_dataframe(
    coalesced: pd.DataFrame,
    *,
    projection: CohortRetryBuilderProjection,
    internal_columns: set[str],
) -> pd.DataFrame:
    """Project a schema-bearing empty canonical result through configured drop semantics."""
    empty = coalesced.head(0).copy()
    drop_patterns: list[str] = list(projection.original_dropped_names)
    for processor in projection.original_config.processors or []:
        if processor.processor_type == ProcessorType.DROP_COLUMNS:
            drop_patterns.extend(processor.column_names)
    to_drop = {
        column
        for column in empty.columns
        if column in internal_columns or any(fnmatch(column, pattern) for pattern in drop_patterns)
    }
    return empty.drop(columns=sorted(to_drop), errors="ignore")


def builder_from_projection(projection: CohortRetryBuilderProjection) -> DataDesignerConfigBuilder:
    """Reconstruct the user's original builder from a retry projection snapshot."""
    return DataDesignerConfigBuilder.from_config(
        BuilderConfig(data_designer=projection.original_config.model_copy(deep=True))
    )


def write_original_builder_config(stage_path: Path, projection: CohortRetryBuilderProjection) -> None:
    """Persist the original declarative builder config at the canonical stage root."""
    BuilderConfig(data_designer=projection.original_config.model_copy(deep=True)).to_json(
        stage_path / SDG_CONFIG_FILENAME
    )


def count_storage_records(storage: ArtifactStorage) -> int:
    """Count rows in an artifact storage's canonical parquet batches."""
    return sum(lazy.pq.read_metadata(path).num_rows for path in storage.final_dataset_path.glob("batch_*.parquet"))


def write_canonical_metadata(
    *,
    storage: ArtifactStorage,
    projection: CohortRetryBuilderProjection,
    manifest: RetryManifest,
    actual_records: int,
    model_usage: dict[str, dict[str, Any]],
) -> None:
    """Restore logical retry identity and aggregate usage in canonical metadata."""
    try:
        metadata = storage.read_metadata()
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise DataDesignerWorkflowError(
            "Final cohort-retry metadata is missing or invalid after durable completion."
        ) from exc
    if not metadata.get("column_statistics"):
        raise DataDesignerWorkflowError("Final cohort-retry profiling metadata is missing after durable completion.")
    storage.write_metadata(
        {
            **metadata,
            **projection.original_config.fingerprint(),
            "target_num_records": manifest.target_records,
            "original_target_num_records": manifest.target_records,
            "actual_num_records": actual_records,
            "cohort_retry": metadata_retry_summary(manifest, model_usage),
        }
    )


def aggregate_model_usage(manifest: RetryManifest) -> dict[str, dict[str, Any]]:
    """Aggregate model, request, tool, and image usage across every durable retry phase."""
    aggregate: dict[str, ModelUsageStats] = {}
    snapshots = [
        manifest.base_model_usage,
        *(attempt.model_usage for attempt in manifest.attempts),
        manifest.final_model_usage,
    ]
    for snapshot in snapshots:
        for model_name, payload in snapshot.items():
            try:
                incoming = ModelUsageStats.model_validate(payload)
            except ValidationError as exc:
                raise DataDesignerWorkflowError(
                    f"Cohort-retry model usage for {model_name!r} is invalid: {exc}"
                ) from exc
            current = aggregate.setdefault(model_name, ModelUsageStats())
            current.extend(
                token_usage=incoming.token_usage,
                request_usage=incoming.request_usage,
                tool_usage=incoming.tool_usage,
                image_usage=incoming.image_usage,
            )
    return {name: usage.model_dump(mode="json") for name, usage in aggregate.items()}


def manifest_summary(manifest: RetryManifest) -> dict[str, Any]:
    """Build the compact retry summary exposed through workflow metadata."""
    sampler_retry_mode = SamplerRetryMode(manifest.policy["sampler_retry_mode"])
    return {
        "target_records": manifest.target_records,
        "accepted_records": manifest.accepted_records,
        "unresolved_records": len(manifest.unresolved_slot_ids),
        "unresolved_slot_ids": manifest.unresolved_slot_ids,
        "candidate_records": manifest.candidate_records,
        "attempts": len(manifest.attempts),
        "sampler_retry_mode": sampler_retry_mode.value,
        "exhausted": bool(manifest.unresolved_slot_ids),
        "distribution_warning": (distribution_warning(sampler_retry_mode) if manifest.unresolved_slot_ids else None),
    }


def metadata_retry_summary(
    manifest: RetryManifest,
    model_usage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete durable retry summary stored in dataset metadata."""
    return {
        **manifest_summary(manifest),
        "seed_column_names": manifest.base_seed_column_names,
        "model_usage": model_usage,
    }


def distribution_warning(mode: SamplerRetryMode) -> str:
    """Describe the distribution bias of a bounded partial result."""
    if mode == SamplerRetryMode.PRESERVE:
        return (
            "The partial result omits unresolved seed/sampler slots and is biased toward cohort combinations "
            "that passed within the retry bounds."
        )
    return (
        "The partial result omits unresolved seed slots and accepted sampler values are additionally conditioned "
        "on passing the predicate."
    )


def clear_ambiguous_finalization(stage_path: Path) -> None:
    """Remove only derived final artifacts left by an interrupted terminal processor pass."""
    metadata_path = stage_path / METADATA_FILENAME
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    if metadata.get("post_generation_state") != "started":
        return
    for name in (
        FINAL_DATASET_FOLDER_NAME,
        _PARTIAL_RESULTS_FOLDER_NAME,
        _DROPPED_COLUMNS_FOLDER_NAME,
        PROCESSORS_OUTPUTS_FOLDER_NAME,
        METADATA_FILENAME,
        SDG_CONFIG_FILENAME,
    ):
        path = stage_path / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def unique_name(base_name: str, used_names: set[str]) -> str:
    """Return a deterministic unused name based on the supplied base."""
    if base_name not in used_names:
        return base_name
    suffix = 1
    while f"{base_name}_{suffix}" in used_names:
        suffix += 1
    return f"{base_name}_{suffix}"
