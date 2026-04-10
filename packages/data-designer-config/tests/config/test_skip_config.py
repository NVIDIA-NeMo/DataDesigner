# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_designer.config.base import SkipConfig
from data_designer.config.column_configs import (
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SeedDatasetColumnConfig,
)
from data_designer.config.sampler_params import SamplerType, UUIDSamplerParams

_BASE_LLM = dict(name="test", prompt="test {{ x }}", model_alias="default")


@pytest.mark.parametrize(
    "value",
    [True, False, 0, 42, 1.5, "skipped", None],
)
def test_skip_config_value_types(value: bool | int | float | str | None) -> None:
    cfg = SkipConfig(when="{{ x == 0 }}", value=value)
    assert cfg.value == value


@pytest.mark.parametrize(
    ("when", "match"),
    [
        pytest.param("{{ 1 + }}", "unexpected", id="syntax-error"),
        pytest.param("in_stock == 0", "does not reference any columns", id="no-delimiters"),
    ],
)
def test_skip_config_when_rejects_invalid_expressions(when: str, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        SkipConfig(when=when)


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        pytest.param("{{ in_stock == 0 }}", ["in_stock"], id="single"),
        pytest.param("{{ a > 0 and b < 10 }}", ["a", "b"], id="multiple"),
    ],
)
def test_skip_config_columns_extraction(when: str, expected: list[str]) -> None:
    cfg = SkipConfig(when=when)
    assert cfg.columns == expected


def test_skip_config_columns_cached() -> None:
    cfg = SkipConfig(when="{{ x == 0 }}")
    first = cfg.columns
    assert cfg.columns is first


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        pytest.param("skip", None, id="skip-defaults-none"),
        pytest.param("propagate_skip", True, id="propagate-skip-defaults-true"),
    ],
)
def test_single_column_config_skip_defaults(attr: str, expected: object) -> None:
    col = LLMTextColumnConfig(**_BASE_LLM)
    assert getattr(col, attr) == expected


def test_skip_rejected_on_sampler_type() -> None:
    with pytest.raises(ValidationError, match="skip is not supported on sampler columns"):
        SamplerColumnConfig(
            name="s",
            sampler_type=SamplerType.UUID,
            params=UUIDSamplerParams(prefix="p_", short_form=True),
            skip=SkipConfig(when="{{ y == 1 }}"),
        )


def test_skip_rejected_on_seed_dataset_type() -> None:
    with pytest.raises(ValidationError, match="skip is not supported on seed-dataset columns"):
        SeedDatasetColumnConfig(
            name="seed_col",
            skip=SkipConfig(when="{{ y == 1 }}"),
        )


def test_skip_rejected_with_allow_resize() -> None:
    with pytest.raises(ValidationError, match="skip and allow_resize cannot be used together"):
        LLMTextColumnConfig(
            **_BASE_LLM,
            allow_resize=True,
            skip=SkipConfig(when="{{ x == 0 }}"),
        )


def test_skip_self_reference_rejected() -> None:
    with pytest.raises(ValidationError, match="references itself"):
        LLMTextColumnConfig(
            name="foo",
            prompt="test {{ bar }}",
            model_alias="default",
            skip=SkipConfig(when="{{ foo == 0 }}"),
        )


def test_skip_side_effect_self_reference_rejected() -> None:
    """Referencing a column's own side-effect (e.g. trace) in skip.when is a self-reference."""
    from data_designer.config.column_configs import TraceType

    with pytest.raises(ValidationError, match="references itself"):
        LLMTextColumnConfig(
            name="review",
            prompt="test {{ bar }}",
            model_alias="default",
            with_trace=TraceType.ALL_MESSAGES,
            skip=SkipConfig(when="{{ review__trace == 'x' }}"),
        )
