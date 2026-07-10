# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

import data_designer.config as dd
from data_designer.config.record_selection import RecordSelectionConfig, RecordSelectionExhaustion


def test_record_selection_config_defaults_to_raise() -> None:
    config = RecordSelectionConfig(predicate_column="accepted", max_candidate_records=100)

    assert config.predicate_column == "accepted"
    assert config.max_candidate_records == 100
    assert RecordSelectionExhaustion(config.on_exhausted) is RecordSelectionExhaustion.RAISE


def test_record_selection_config_coerces_exhaustion_string() -> None:
    config = RecordSelectionConfig(
        predicate_column="accepted",
        max_candidate_records=100,
        on_exhausted="return_partial",
    )

    assert RecordSelectionExhaustion(config.on_exhausted) is RecordSelectionExhaustion.RETURN_PARTIAL


@pytest.mark.parametrize("max_candidate_records", [0, -1, True, 1.0, "1"])
def test_record_selection_config_rejects_invalid_candidate_bound(max_candidate_records: object) -> None:
    with pytest.raises(ValidationError, match="max_candidate_records"):
        RecordSelectionConfig(
            predicate_column="accepted",
            max_candidate_records=max_candidate_records,
        )


@pytest.mark.parametrize("predicate_column", ["", "   "])
def test_record_selection_config_rejects_blank_predicate_column(predicate_column: str) -> None:
    with pytest.raises(ValidationError, match="predicate_column"):
        RecordSelectionConfig(
            predicate_column=predicate_column,
            max_candidate_records=100,
        )


def test_record_selection_config_rejects_unknown_exhaustion_behavior() -> None:
    with pytest.raises(ValidationError, match="on_exhausted"):
        RecordSelectionConfig(
            predicate_column="accepted",
            max_candidate_records=100,
            on_exhausted="retry_forever",
        )


def test_record_selection_config_round_trips_serialization() -> None:
    config = RecordSelectionConfig(
        predicate_column="accepted",
        max_candidate_records=100,
        on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
    )

    serialized = config.model_dump(mode="json")

    assert serialized == {
        "predicate_column": "accepted",
        "max_candidate_records": 100,
        "on_exhausted": "return_partial",
    }
    assert RecordSelectionConfig.model_validate(serialized) == config


def test_record_selection_types_are_exported_from_config_package() -> None:
    assert dd.RecordSelectionConfig is RecordSelectionConfig
    assert dd.RecordSelectionExhaustion is RecordSelectionExhaustion
