# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from data_designer.config.analysis.column_profilers import JudgeScoreProfilerConfig
from data_designer.config.base import SkipConfig
from data_designer.config.column_configs import (
    CustomColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.fingerprint import (
    CONFIG_HASH_ALGO,
    CONFIG_HASH_VERSION,
    fingerprint_config,
)
from data_designer.config.mcp import ToolConfig
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.record_selection import RecordSelectionConfig, RecordSelectionExhaustion
from data_designer.config.sampler_constraints import InequalityOperator, ScalarInequalityConstraint
from data_designer.config.sampler_params import CategorySamplerParams, UniformSamplerParams
from data_designer.config.seed import IndexRange, SamplingStrategy, SeedConfig
from data_designer.config.seed_source import HuggingFaceSeedSource


def _compute_hash(config: DataDesignerConfig) -> str:
    return str(fingerprint_config(config)["config_hash"])


def test_fingerprint_shape(stub_data_designer_config: DataDesignerConfig) -> None:
    fp = stub_data_designer_config.fingerprint()
    assert set(fp.keys()) == {"config_hash", "config_hash_algo", "config_hash_version"}
    assert fp["config_hash_algo"] == CONFIG_HASH_ALGO
    assert fp["config_hash_version"] == CONFIG_HASH_VERSION
    assert fp["config_hash"].startswith(f"{CONFIG_HASH_ALGO}:")
    digest = fp["config_hash"].split(":", 1)[1]
    assert len(digest) == 64  # sha256 hex
    assert all(c in "0123456789abcdef" for c in digest)


def test_fingerprint_deterministic_within_process(
    stub_data_designer_config: DataDesignerConfig,
    stub_data_designer_config_str: str,
) -> None:
    rebuilt = DataDesignerConfig.model_validate(yaml.safe_load(stub_data_designer_config_str))
    assert _compute_hash(stub_data_designer_config) == _compute_hash(rebuilt)


def test_fingerprint_deterministic_across_processes(stub_data_designer_config_str: str) -> None:
    """A separate Python process must produce the same digest for the same config."""
    script = f"""
import sys, yaml
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.fingerprint import fingerprint_config

cfg = DataDesignerConfig.model_validate(yaml.safe_load({stub_data_designer_config_str!r}))
sys.stdout.write(fingerprint_config(cfg)["config_hash"])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    out = result.stdout.strip()

    cfg = DataDesignerConfig.model_validate(yaml.safe_load(stub_data_designer_config_str))
    assert out == _compute_hash(cfg)


# ---------------------------------------------------------------------------
# Helpers for building minimal configs in include/exclude tests.
# ---------------------------------------------------------------------------


def _make_model(alias: str = "m", model: str = "some-model", provider: str = "some-provider") -> ModelConfig:
    return ModelConfig(
        alias=alias,
        model=model,
        provider=provider,
        inference_parameters=ChatCompletionInferenceParams(temperature=0.5, top_p=0.9, max_tokens=128),
    )


def _make_minimal_config(**overrides: object) -> DataDesignerConfig:
    base: dict[str, Any] = {
        "columns": [SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1))],
        "model_configs": [_make_model()],
    }
    base.update(overrides)
    return DataDesignerConfig(**base)


# ---------------------------------------------------------------------------
# INCLUDE: identity-relevant changes must change the hash.
# ---------------------------------------------------------------------------


def test_changing_column_name_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        columns=[SamplerColumnConfig(name="y", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1))],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_column_type_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        columns=[
            SamplerColumnConfig(name="x", sampler_type="category", params=CategorySamplerParams(values=["a", "b"])),
        ],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_sampler_params_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        columns=[SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=2))],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_model_identity_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(model_configs=[ModelConfig(alias="m", model="other-model", provider="some-provider")])
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_temperature_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                provider="some-provider",
                inference_parameters=ChatCompletionInferenceParams(temperature=0.99, top_p=0.9, max_tokens=128),
            )
        ],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_column_order_changes_hash() -> None:
    """Column order is part of identity (DAG ordering)."""
    cols_a = [
        SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1)),
        SamplerColumnConfig(name="y", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1)),
    ]
    cols_b = list(reversed(cols_a))
    assert _compute_hash(_make_minimal_config(columns=cols_a)) != _compute_hash(_make_minimal_config(columns=cols_b))


def test_changing_skip_changes_hash() -> None:
    base_col = LLMTextColumnConfig(name="t", prompt="hi {{x}}", model_alias="m")
    skipped = LLMTextColumnConfig(
        name="t",
        prompt="hi {{x}}",
        model_alias="m",
        skip=SkipConfig(when="{{ x > 0 }}"),
    )
    cols_no_skip = [
        SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1)),
        base_col,
    ]
    cols_skip = [
        SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1)),
        skipped,
    ]
    assert _compute_hash(_make_minimal_config(columns=cols_no_skip)) != _compute_hash(
        _make_minimal_config(columns=cols_skip)
    )


def test_changing_constraint_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        constraints=[ScalarInequalityConstraint(target_column="x", operator=InequalityOperator.LT, rhs=0.5)],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_top_level_processor_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(processors=[DropColumnsProcessorConfig(name="drop", column_names=["x"])])
    assert _compute_hash(a) != _compute_hash(b)


def test_adding_record_selection_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        record_selection=RecordSelectionConfig(predicate_column="accepted", max_candidate_records=100)
    )
    assert _compute_hash(a) != _compute_hash(b)


@pytest.mark.parametrize(
    "record_selection",
    [
        RecordSelectionConfig(predicate_column="other_predicate", max_candidate_records=100),
        RecordSelectionConfig(predicate_column="accepted", max_candidate_records=200),
        RecordSelectionConfig(
            predicate_column="accepted",
            max_candidate_records=100,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        ),
    ],
)
def test_changing_record_selection_field_changes_hash(record_selection: RecordSelectionConfig) -> None:
    baseline = RecordSelectionConfig(predicate_column="accepted", max_candidate_records=100)

    assert _compute_hash(_make_minimal_config(record_selection=baseline)) != _compute_hash(
        _make_minimal_config(record_selection=record_selection)
    )


def test_changing_extra_body_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                provider="some-provider",
                inference_parameters=ChatCompletionInferenceParams(
                    temperature=0.5, top_p=0.9, max_tokens=128, extra_body={"frequency_penalty": 0.5}
                ),
            )
        ],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_provider_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                provider="custom-provider",
                inference_parameters=ChatCompletionInferenceParams(temperature=0.5, top_p=0.9, max_tokens=128),
            )
        ],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_sampling_strategy_changes_hash() -> None:
    a = _make_minimal_config(
        seed_config=SeedConfig(
            source=HuggingFaceSeedSource(path="datasets/x/y/data.csv"),
            sampling_strategy=SamplingStrategy.ORDERED,
        ),
    )
    b = _make_minimal_config(
        seed_config=SeedConfig(
            source=HuggingFaceSeedSource(path="datasets/x/y/data.csv"),
            sampling_strategy=SamplingStrategy.SHUFFLE,
        ),
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_selection_strategy_changes_hash() -> None:
    a = _make_minimal_config(
        seed_config=SeedConfig(source=HuggingFaceSeedSource(path="datasets/x/y/data.csv")),
    )
    b = _make_minimal_config(
        seed_config=SeedConfig(
            source=HuggingFaceSeedSource(path="datasets/x/y/data.csv"),
            selection_strategy=IndexRange(start=0, end=99),
        ),
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_adding_tool_config_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t", providers=["p"])])
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_tool_config_alias_changes_hash() -> None:
    a = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t1", providers=["p"])])
    b = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t2", providers=["p"])])
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_tool_config_providers_changes_hash() -> None:
    a = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t", providers=["p1"])])
    b = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t", providers=["p2"])])
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_tool_config_allow_tools_changes_hash() -> None:
    a = _make_minimal_config(
        tool_configs=[ToolConfig(tool_alias="t", providers=["p"], allow_tools=["search"])],
    )
    b = _make_minimal_config(
        tool_configs=[ToolConfig(tool_alias="t", providers=["p"], allow_tools=["search", "list"])],
    )
    assert _compute_hash(a) != _compute_hash(b)


def test_changing_max_tool_call_turns_changes_hash() -> None:
    a = _make_minimal_config(
        tool_configs=[ToolConfig(tool_alias="t", providers=["p"], max_tool_call_turns=5)],
    )
    b = _make_minimal_config(
        tool_configs=[ToolConfig(tool_alias="t", providers=["p"], max_tool_call_turns=10)],
    )
    assert _compute_hash(a) != _compute_hash(b)


# ---------------------------------------------------------------------------
# EXCLUDE: non-identity changes must NOT change the hash.
# ---------------------------------------------------------------------------


def test_skip_health_check_does_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                provider="some-provider",
                inference_parameters=ChatCompletionInferenceParams(temperature=0.5, top_p=0.9, max_tokens=128),
                skip_health_check=True,
            )
        ],
    )
    assert _compute_hash(a) == _compute_hash(b)


def test_max_parallel_requests_does_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                provider="some-provider",
                inference_parameters=ChatCompletionInferenceParams(
                    temperature=0.5, top_p=0.9, max_tokens=128, max_parallel_requests=32
                ),
            )
        ],
    )
    assert _compute_hash(a) == _compute_hash(b)


def test_inference_timeout_does_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                provider="some-provider",
                inference_parameters=ChatCompletionInferenceParams(
                    temperature=0.5, top_p=0.9, max_tokens=128, timeout=30
                ),
            )
        ],
    )
    assert _compute_hash(a) == _compute_hash(b)


def test_tool_config_timeout_sec_does_not_change_hash() -> None:
    a = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t", providers=["p"])])
    b = _make_minimal_config(
        tool_configs=[ToolConfig(tool_alias="t", providers=["p"], timeout_sec=30.0)],
    )
    assert _compute_hash(a) == _compute_hash(b)


def test_profilers_do_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(profilers=[JudgeScoreProfilerConfig(model_alias="m")])
    assert _compute_hash(a) == _compute_hash(b)


def test_hf_seed_token_and_endpoint_do_not_change_hash() -> None:
    a = _make_minimal_config(
        seed_config=SeedConfig(source=HuggingFaceSeedSource(path="datasets/x/y/data.csv")),
    )
    b = _make_minimal_config(
        seed_config=SeedConfig(
            source=HuggingFaceSeedSource(
                path="datasets/x/y/data.csv",
                token="secret",
                endpoint="https://example.com",
            ),
        ),
    )
    assert _compute_hash(a) == _compute_hash(b)


def test_changing_hf_seed_path_changes_hash() -> None:
    a = _make_minimal_config(seed_config=SeedConfig(source=HuggingFaceSeedSource(path="datasets/x/y/a.csv")))
    b = _make_minimal_config(seed_config=SeedConfig(source=HuggingFaceSeedSource(path="datasets/x/y/b.csv")))
    assert _compute_hash(a) != _compute_hash(b)


# ---------------------------------------------------------------------------
# Canonicalization: alias-keyed lookup tables are order-independent, and
# `None`/empty-list optional collections collapse to a single representation.
# ---------------------------------------------------------------------------


def test_model_configs_order_independent() -> None:
    """`model_configs` is alias-keyed; reordering the list must not flip the hash."""
    a = _make_minimal_config(model_configs=[_make_model("m1"), _make_model("m2")])
    b = _make_minimal_config(model_configs=[_make_model("m2"), _make_model("m1")])
    assert _compute_hash(a) == _compute_hash(b)


def test_tool_configs_order_independent() -> None:
    """`tool_configs` is alias-keyed; reordering the list must not flip the hash."""
    a = _make_minimal_config(
        tool_configs=[
            ToolConfig(tool_alias="t1", providers=["p"]),
            ToolConfig(tool_alias="t2", providers=["p"]),
        ],
    )
    b = _make_minimal_config(
        tool_configs=[
            ToolConfig(tool_alias="t2", providers=["p"]),
            ToolConfig(tool_alias="t1", providers=["p"]),
        ],
    )
    assert _compute_hash(a) == _compute_hash(b)


@pytest.mark.parametrize(
    "field",
    ["model_configs", "tool_configs", "constraints", "processors"],
)
def test_none_vs_empty_list_for_optional_top_level_fields_match(field: str) -> None:
    """`None` and `[]` must produce identical hashes for optional top-level collections."""
    base_kwargs: dict[str, Any] = {
        "columns": [SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1))],
    }
    a = DataDesignerConfig(**base_kwargs, **{field: None})
    b = DataDesignerConfig(**base_kwargs, **{field: []})
    assert _compute_hash(a) == _compute_hash(b)


def test_tool_config_allow_tools_none_vs_empty_match() -> None:
    """`allow_tools=None` and `allow_tools=[]` must produce identical hashes."""
    a = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t", providers=["p"], allow_tools=None)])
    b = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t", providers=["p"], allow_tools=[])])
    assert _compute_hash(a) == _compute_hash(b)


# ---------------------------------------------------------------------------
# Custom column identity: name + qualname + module + decorator metadata.
# ---------------------------------------------------------------------------


class _GenParamsV1(BaseModel):
    factor: int = 1


@custom_column_generator()
def _generate_v1(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover - logic not exercised
    return str(row.get("x", 0) * generator_params.factor)


@custom_column_generator()
def _generate_v2(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover - logic not exercised
    return str(row.get("x", 0) * generator_params.factor + 1)


def _make_custom_config(fn: Any, params: _GenParamsV1 | None = None) -> DataDesignerConfig:
    return _make_minimal_config(
        columns=[
            SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1)),
            CustomColumnConfig(
                name="c",
                generator_function=fn,
                generator_params=params or _GenParamsV1(),
            ),
        ],
    )


def test_custom_column_includes_generator_params() -> None:
    a = _make_custom_config(_generate_v1, _GenParamsV1(factor=1))
    b = _make_custom_config(_generate_v1, _GenParamsV1(factor=2))
    assert _compute_hash(a) != _compute_hash(b)


def test_custom_column_includes_generator_function_name() -> None:
    a = _make_custom_config(_generate_v1)
    b = _make_custom_config(_generate_v2)
    assert _compute_hash(a) != _compute_hash(b)


def test_custom_column_qualname_disambiguates_same_name() -> None:
    """Two functions sharing `__name__` but with different `__qualname__` must
    produce different hashes (the fix for the same-name-different-scope
    collision class)."""

    def _make_outer_a() -> Any:
        @custom_column_generator()
        def _gen(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover
            return ""

        return _gen

    def _make_outer_b() -> Any:
        @custom_column_generator()
        def _gen(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover
            return ""

        return _gen

    fn_a = _make_outer_a()
    fn_b = _make_outer_b()
    assert fn_a.__name__ == fn_b.__name__
    assert fn_a.__qualname__ != fn_b.__qualname__

    assert _compute_hash(_make_custom_config(fn_a)) != _compute_hash(_make_custom_config(fn_b))


def test_custom_column_decorator_metadata_changes_hash() -> None:
    """Two generators sharing `__name__` and `__qualname__` but with different
    `@custom_column_generator()` metadata (`required_columns` etc.) must
    produce different hashes — `required_columns` changes DAG order and
    `side_effect_columns` changes the output schema."""

    def _make_with_required(required_cols: list[str]) -> Any:
        @custom_column_generator(required_columns=required_cols)
        def _gen(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover
            return ""

        return _gen

    fn_a = _make_with_required(["x"])
    fn_b = _make_with_required(["x", "y"])
    assert fn_a.__name__ == fn_b.__name__
    assert fn_a.__qualname__ == fn_b.__qualname__
    assert fn_a.custom_column_metadata != fn_b.custom_column_metadata

    assert _compute_hash(_make_custom_config(fn_a)) != _compute_hash(_make_custom_config(fn_b))


def test_closure_captured_state_is_a_known_limitation() -> None:
    """Pin the documented closure-capture limitation.

    Factory-built closures with different captured state share `__name__`,
    `__qualname__`, `__module__`, and source, so they fingerprint identically.
    If this test ever flips, update or remove the closure-capture Limitation
    block in `fingerprint_config()`'s docstring (and the matching note in the
    PR description / public docs) so the contract and the implementation stay
    in sync.
    """

    def _make_factor_gen(factor: int) -> Any:
        @custom_column_generator()
        def _gen(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover
            return str(row.get("x", 0) * factor)

        return _gen

    fn_a = _make_factor_gen(2)
    fn_b = _make_factor_gen(7)
    assert fn_a.__name__ == fn_b.__name__
    assert fn_a.__qualname__ == fn_b.__qualname__

    assert _compute_hash(_make_custom_config(fn_a)) == _compute_hash(_make_custom_config(fn_b))
