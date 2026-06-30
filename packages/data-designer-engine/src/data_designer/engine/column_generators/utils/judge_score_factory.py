# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from data_designer.config.column_configs import Score

SCORING_FORMAT = "* {score}: {description}"
SCORE_FIELD_DESCRIPTION_FORMAT = "Score Descriptions for {enum_name}:\n{scoring}"


def _normalize_score_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().casefold()


def _coerce_score_value(value: Any, enum_type: type[Enum]) -> Any:
    for member in enum_type:
        if isinstance(value, bool) != isinstance(member.value, bool):
            continue
        if value == member.value:
            return value

    normalized_value = _normalize_score_value(value)
    matches = [member.value for member in enum_type if _normalize_score_value(member.value) == normalized_value]
    if len(matches) == 1:
        return matches[0]
    return value


class BaseJudgeResponse(BaseModel):
    """Base model for all rubrics."""

    model_config = ConfigDict(use_enum_values=True)
    reasoning: str = Field(..., description="Reasoning for the assigned score.")

    @model_validator(mode="before")
    @classmethod
    def coerce_score(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "score" not in data:
            return data

        score_field = cls.model_fields.get("score")
        if score_field is None:
            return data

        score_type = score_field.annotation
        if not isinstance(score_type, type) or not issubclass(score_type, Enum):
            return data

        coerced_data = data.copy()
        coerced_data["score"] = _coerce_score_value(data["score"], score_type)
        return coerced_data


def _stringify_scoring(options: dict, enum_type: type[Enum]) -> str:
    """Convert score descriptions into a single text block."""
    list_block = "\n".join(
        [SCORING_FORMAT.format(score=score, description=description) for score, description in options.items()]
    )
    return SCORE_FIELD_DESCRIPTION_FORMAT.format(enum_name=enum_type.__name__, scoring=list_block)


def create_judge_response_model(score: Score) -> type[BaseJudgeResponse]:
    """Create a JudgeResponse data type."""
    enum_members = {}
    for option in score.options.keys():
        member_name = f"VALUE_{option}"
        enum_members[member_name] = option

    DynamicScaleEnum = Enum(f"{score.name}Enum", enum_members)
    options = _stringify_scoring(score.options, enum_type=DynamicScaleEnum)

    return create_model(
        score.name,
        __doc__=score.description if score.description else None,
        __base__=BaseJudgeResponse,
        score=(DynamicScaleEnum, Field(..., description=options)),
    )


def create_judge_structured_output_model(
    judge_responses: list[type[BaseJudgeResponse]],
) -> type[BaseModel]:
    """Create a JudgeStructuredOutput class dynamically."""
    return create_model(
        "JudgeStructuredOutput",
        __doc__=f"Response schema for scores with the following names: {[response.__name__ for response in judge_responses]}.",
        __base__=BaseModel,
        **{response.__name__: (response, ...) for response in judge_responses},
    )
