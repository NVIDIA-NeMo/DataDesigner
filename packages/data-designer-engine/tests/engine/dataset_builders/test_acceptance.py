# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_designer.config.record_selection import RecordSelectionConfig
from data_designer.engine.dataset_builders.acceptance import (
    AcceptanceController,
    SelectionBatchMarker,
)
from data_designer.engine.dataset_builders.errors import DatasetGenerationError


def _controller(*, target: int = 3, cap: int = 10, buffer_size: int = 4) -> AcceptanceController:
    return AcceptanceController(
        config=RecordSelectionConfig(predicate_column="keep", max_candidate_records=cap),
        target_records=target,
        buffer_size=buffer_size,
    )


def test_next_candidate_batch_tracks_candidate_coordinate_space() -> None:
    controller = _controller(target=3, cap=8, buffer_size=5)
    first = controller.next_candidate_batch()
    assert (first.candidate_batch_id, first.start_offset, first.size) == (0, 0, 3)

    decision = controller.select(pd.DataFrame({"keep": [False, True, False]}), batch=first, failed_generation_records=0)
    controller.record_checkpoint(
        batch=first, decision=decision, accepted_partition="selection-accepted/batch_00000.parquet"
    )

    second = controller.next_candidate_batch()
    assert (second.candidate_batch_id, second.start_offset, second.size) == (1, 3, 3)


def test_select_accepts_python_and_numpy_booleans_and_rejects_nulls() -> None:
    controller = _controller(target=3)
    batch = controller.next_candidate_batch()
    decision = controller.select(
        pd.DataFrame({"keep": [True, np.bool_(True), None]}),
        batch=batch,
        failed_generation_records=0,
    )
    assert decision.accepted_indices == (0, 1)
    assert decision.accepted_records == 2
    assert decision.null_predicate_records == 1


def test_select_trims_earliest_accepted_rows() -> None:
    controller = _controller(target=2, buffer_size=5)
    first = controller.next_candidate_batch()
    first_decision = controller.select(pd.DataFrame({"keep": [True, False]}), batch=first, failed_generation_records=0)
    controller.record_checkpoint(
        batch=first,
        decision=first_decision,
        accepted_partition="selection-accepted/batch_00000.parquet",
    )

    second = controller.next_candidate_batch()
    second_decision = controller.select(pd.DataFrame({"keep": [True, True]}), batch=second, failed_generation_records=0)
    assert second_decision.accepted_indices == (0,)
    assert second_decision.trimmed_accepted_records == 1


def test_select_accounts_for_failed_generation_slots() -> None:
    controller = _controller(target=3, buffer_size=3)
    batch = controller.next_candidate_batch()
    decision = controller.select(pd.DataFrame({"keep": [True, False]}), batch=batch, failed_generation_records=1)
    assert decision.failed_generation_records == 1
    assert decision.accepted_records == 1
    assert decision.rejected_records == 1


@pytest.mark.parametrize("value", [1, 0, "true", "false", [], {}])
def test_select_rejects_non_boolean_predicate_values(value: object) -> None:
    controller = _controller(target=1, cap=1, buffer_size=1)
    batch = controller.next_candidate_batch()
    with pytest.raises(DatasetGenerationError, match="non-boolean"):
        controller.select(pd.DataFrame({"keep": [value]}), batch=batch, failed_generation_records=0)


def test_controller_rejects_candidate_cap_below_target() -> None:
    with pytest.raises(DatasetGenerationError, match="greater than or equal"):
        _controller(target=4, cap=3)


def test_marker_sequence_hydrates_progress() -> None:
    marker = SelectionBatchMarker(
        candidate_batch_id=0,
        row_group_id=0,
        candidate_start_offset=0,
        candidate_records=3,
        accepted_records=1,
        rejected_records=1,
        null_predicate_records=1,
        failed_generation_records=0,
        trimmed_accepted_records=0,
        accepted_partition="selection-accepted/batch_00000.parquet",
    )
    controller = AcceptanceController(
        config=RecordSelectionConfig(predicate_column="keep", max_candidate_records=6),
        target_records=3,
        buffer_size=3,
        markers=(marker,),
    )
    assert controller.candidate_records == 3
    assert controller.accepted_records == 1
    assert controller.next_candidate_batch().start_offset == 3


def test_zero_acceptance_marker_has_no_partition() -> None:
    marker = SelectionBatchMarker(
        candidate_batch_id=0,
        row_group_id=0,
        candidate_start_offset=0,
        candidate_records=2,
        accepted_records=0,
        rejected_records=2,
        null_predicate_records=0,
        failed_generation_records=0,
        trimmed_accepted_records=0,
        accepted_partition=None,
    )
    assert SelectionBatchMarker.from_dict(marker.to_dict()) == marker
