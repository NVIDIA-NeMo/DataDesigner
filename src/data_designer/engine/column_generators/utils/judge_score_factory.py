# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Type

from pydantic import BaseModel, ConfigDict, Field, create_model

from data_designer.config.column_configs import Score

SCORING_FORMAT = "* {score}: {description}"
SCORE_FIELD_DESCRIPTION_FORMAT = "Score Descriptions for {enum_name}:\n{scoring}"


class BaseJudgeResponse(BaseModel):
    """Base model for all rubrics."""

    model_config = ConfigDict(use_enum_values=True)
    reasoning: str = Field(..., description="Reasoning for the assigned score.")


def _stringify_scoring(options: dict, enum_type: Type[Enum]) -> str:
    """Convert score descriptions into a single text block."""
    list_block = "\n".join(
        [SCORING_FORMAT.format(score=score, description=description) for score, description in options.items()]
    )
    return SCORE_FIELD_DESCRIPTION_FORMAT.format(enum_name=enum_type.__name__, scoring=list_block)


def create_judge_response_model(score: Score) -> Type[BaseJudgeResponse]:
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
    judge_responses: list[Type[BaseJudgeResponse]],
) -> Type[BaseModel]:
    """Create a JudgeStructuredOutput class dynamically."""
    return create_model(
        "JudgeStructuredOutput",
        __doc__=f"Response schema for scores with the following names: {[response.__name__ for response in judge_responses]}.",
        __base__=BaseModel,
        **{response.__name__: (response, ...) for response in judge_responses},
    )
