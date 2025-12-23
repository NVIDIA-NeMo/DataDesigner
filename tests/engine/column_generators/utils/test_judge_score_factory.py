# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import pytest
from pydantic import BaseModel

from data_designer.config.column_configs import Score
from data_designer.engine.column_generators.utils.judge_score_factory import (
    SCORE_FIELD_DESCRIPTION_FORMAT,
    SCORING_FORMAT,
    BaseJudgeResponse,
    _stringify_scoring,
    create_judge_response_model,
    create_judge_structured_output_model,
)


def test_judge_score_factory_scoring_constants():
    assert SCORING_FORMAT == "* {score}: {description}"
    assert SCORE_FIELD_DESCRIPTION_FORMAT == "Score Descriptions for {enum_name}:\n{scoring}"


def test_judge_score_factory_base_judge_response():
    response = BaseJudgeResponse(reasoning="Test reasoning")
    assert response.reasoning == "Test reasoning"
    assert response.model_config["use_enum_values"] is True

    with pytest.raises(ValueError):
        BaseJudgeResponse()


def test_judge_score_factory_stringify_scoring():
    class TestEnum(Enum):
        HIGH = "high"
        LOW = "low"

    options = {"high": "High quality", "low": "Low quality"}
    result = _stringify_scoring(options, TestEnum)

    expected = "Score Descriptions for TestEnum:\n* high: High quality\n* low: Low quality"
    assert result == expected


def test_judge_score_factory_create_judge_response_model():
    score = Score(
        name="quality_score",
        description="Quality assessment score",
        options={"high": "High quality", "low": "Low quality"},
    )

    model_class = create_judge_response_model(score)

    assert issubclass(model_class, BaseJudgeResponse)
    assert "score" in model_class.model_fields
    assert "reasoning" in model_class.model_fields

    instance = model_class(score="high", reasoning="Test reasoning")
    assert instance.score == "high"
    assert instance.reasoning == "Test reasoning"


def test_judge_score_factory_create_judge_structured_output_model():
    score = Score(
        name="quality_score",
        description="Quality assessment score",
        options={"high": "High quality", "low": "Low quality"},
    )

    response_model = create_judge_response_model(score)
    model_class = create_judge_structured_output_model([response_model])

    assert issubclass(model_class, BaseModel)
    assert "quality_score" in model_class.model_fields


def test_judge_score_factory_preserves_score_name_casing():
    """Test that Score name casing is preserved in the JSON output keys."""
    score = Score(
        name="Some Name with mixed - Casing",
        description="Some information about the score.",
        options={"5": "Fantastic", "3": "Needs improvement", "1": "Terrible"},
    )

    response_model = create_judge_response_model(score)
    model_class = create_judge_structured_output_model([response_model])

    assert issubclass(model_class, BaseModel)
    assert "Some Name with mixed - Casing" in model_class.model_fields

    instance = model_class(**{"Some Name with mixed - Casing": {"score": "5", "reasoning": "Test reasoning"}})
    assert hasattr(instance, "Some Name with mixed - Casing")
    output_dict = instance.model_dump()
    assert "Some Name with mixed - Casing" in output_dict
    assert output_dict["Some Name with mixed - Casing"]["score"] == "5"
    assert output_dict["Some Name with mixed - Casing"]["reasoning"] == "Test reasoning"
