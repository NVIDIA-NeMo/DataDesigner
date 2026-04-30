# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from data_designer.config import fingerprint as fp_mod
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
from data_designer.config.sampler_constraints import InequalityOperator, ScalarInequalityConstraint
from data_designer.config.sampler_params import CategorySamplerParams, UniformSamplerParams
from data_designer.config.seed import IndexRange, SamplingStrategy, SeedConfig
from data_designer.config.seed_source import HuggingFaceSeedSource


def _hash(config: DataDesignerConfig, *, custom_column_source: bool = False) -> str:
    return str(fingerprint_config(config, custom_column_source=custom_column_source)["config_hash"])


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
    assert _hash(stub_data_designer_config) == _hash(rebuilt)


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
    assert out == _hash(cfg)


# ---------------------------------------------------------------------------
# Helpers for building minimal configs in include/exclude tests.
# ---------------------------------------------------------------------------


def _make_model() -> ModelConfig:
    return ModelConfig(
        alias="m",
        model="some-model",
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
    assert _hash(a) != _hash(b)


def test_changing_column_type_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        columns=[
            SamplerColumnConfig(name="x", sampler_type="category", params=CategorySamplerParams(values=["a", "b"])),
        ],
    )
    assert _hash(a) != _hash(b)


def test_changing_sampler_params_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        columns=[SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=2))],
    )
    assert _hash(a) != _hash(b)


def test_changing_model_identity_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(model_configs=[ModelConfig(alias="m", model="other-model")])
    assert _hash(a) != _hash(b)


def test_changing_temperature_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                inference_parameters=ChatCompletionInferenceParams(temperature=0.99, top_p=0.9, max_tokens=128),
            )
        ],
    )
    assert _hash(a) != _hash(b)


def test_changing_column_order_changes_hash() -> None:
    cols_a = [
        SamplerColumnConfig(name="x", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1)),
        SamplerColumnConfig(name="y", sampler_type="uniform", params=UniformSamplerParams(low=0, high=1)),
    ]
    cols_b = list(reversed(cols_a))
    assert _hash(_make_minimal_config(columns=cols_a)) != _hash(_make_minimal_config(columns=cols_b))


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
    assert _hash(_make_minimal_config(columns=cols_no_skip)) != _hash(_make_minimal_config(columns=cols_skip))


def test_changing_constraint_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        constraints=[ScalarInequalityConstraint(target_column="x", operator=InequalityOperator.LT, rhs=0.5)],
    )
    assert _hash(a) != _hash(b)


def test_changing_top_level_processor_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(processors=[DropColumnsProcessorConfig(name="drop", column_names=["x"])])
    assert _hash(a) != _hash(b)


def test_changing_extra_body_changes_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                inference_parameters=ChatCompletionInferenceParams(
                    temperature=0.5, top_p=0.9, max_tokens=128, extra_body={"frequency_penalty": 0.5}
                ),
            )
        ],
    )
    assert _hash(a) != _hash(b)


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
    assert _hash(a) != _hash(b)


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
    assert _hash(a) != _hash(b)


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
    assert _hash(a) != _hash(b)


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
                inference_parameters=ChatCompletionInferenceParams(temperature=0.5, top_p=0.9, max_tokens=128),
                skip_health_check=True,
            )
        ],
    )
    assert _hash(a) == _hash(b)


def test_max_parallel_requests_does_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                inference_parameters=ChatCompletionInferenceParams(
                    temperature=0.5, top_p=0.9, max_tokens=128, max_parallel_requests=32
                ),
            )
        ],
    )
    assert _hash(a) == _hash(b)


def test_inference_timeout_does_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(
        model_configs=[
            ModelConfig(
                alias="m",
                model="some-model",
                inference_parameters=ChatCompletionInferenceParams(
                    temperature=0.5, top_p=0.9, max_tokens=128, timeout=30
                ),
            )
        ],
    )
    assert _hash(a) == _hash(b)


def test_tool_configs_do_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(tool_configs=[ToolConfig(tool_alias="t", providers=["p"])])
    assert _hash(a) == _hash(b)


def test_profilers_do_not_change_hash() -> None:
    a = _make_minimal_config()
    b = _make_minimal_config(profilers=[JudgeScoreProfilerConfig(model_alias="m")])
    assert _hash(a) == _hash(b)


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
    assert _hash(a) == _hash(b)


def test_changing_hf_seed_path_changes_hash() -> None:
    a = _make_minimal_config(seed_config=SeedConfig(source=HuggingFaceSeedSource(path="datasets/x/y/a.csv")))
    b = _make_minimal_config(seed_config=SeedConfig(source=HuggingFaceSeedSource(path="datasets/x/y/b.csv")))
    assert _hash(a) != _hash(b)


# ---------------------------------------------------------------------------
# Custom columns: L1 (default) and L2 (opt-in source hashing).
# ---------------------------------------------------------------------------


class _GenParamsV1(BaseModel):
    factor: int = 1


@custom_column_generator()
def _generate_v1(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover - logic not exercised
    return str(row.get("x", 0) * generator_params.factor)


@custom_column_generator()
def _generate_v2(row: dict, generator_params: _GenParamsV1) -> str:  # pragma: no cover - logic not exercised
    return str(row.get("x", 0) * generator_params.factor + 1)


def _make_custom_config(fn: Callable[..., Any], params: _GenParamsV1 | None = None) -> DataDesignerConfig:
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


def test_custom_column_l1_includes_generator_params() -> None:
    a = _make_custom_config(_generate_v1, _GenParamsV1(factor=1))
    b = _make_custom_config(_generate_v1, _GenParamsV1(factor=2))
    assert _hash(a) != _hash(b)


def test_custom_column_l1_includes_generator_function_name() -> None:
    a = _make_custom_config(_generate_v1)
    b = _make_custom_config(_generate_v2)
    # Different function names serialize to different values via field_serializer.
    assert _hash(a) != _hash(b)


def test_custom_column_l2_detects_source_change(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_custom_config(_generate_v1)
    base = _hash(a, custom_column_source=True)

    # Simulate an implementation edit by feeding a different source string.
    sources = iter(["original-source", "edited-source"])
    monkeypatch.setattr(fp_mod, "_hash_custom_column_source", lambda fn, name: next(sources))

    edit_first = _hash(a, custom_column_source=True)
    edit_second = _hash(a, custom_column_source=True)
    assert edit_first != edit_second
    assert base != edit_first


def test_custom_column_unhashable_source_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A plugin whose source can't be retrieved should warn, not raise."""

    def _raise_oserror(_fn: object) -> str:
        raise OSError("compiled / zipped plugin")

    monkeypatch.setattr(inspect, "getsource", _raise_oserror)

    a = _make_custom_config(_generate_v1)
    with caplog.at_level("WARNING", logger=fp_mod.__name__):
        out = a.fingerprint(custom_column_source=True)
    assert out["config_hash"].startswith(f"{CONFIG_HASH_ALGO}:")
    assert "Could not retrieve source" in caplog.text


def test_l1_and_l2_produce_different_hashes() -> None:
    a = _make_custom_config(_generate_v1)
    assert _hash(a) != _hash(a, custom_column_source=True)
