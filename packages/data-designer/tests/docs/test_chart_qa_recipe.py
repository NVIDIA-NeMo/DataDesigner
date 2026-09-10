# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_PATH = REPO_ROOT / "docs" / "assets" / "recipes" / "chart_qa" / "chart_qa.py"


def _load_recipe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("chart_qa_recipe", RECIPE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipe() -> ModuleType:
    return _load_recipe()


@pytest.fixture
def seed_path(tmp_path: Path) -> Path:
    path = tmp_path / "chart_seed.jsonl"
    path.write_text('{"image_url": "https://example.com/chart.png"}\n', encoding="utf-8")
    return path


@pytest.mark.parametrize("prompt_version", ["level1", "level2"])
def test_chart_prompt_versions_match_pipeline_config(recipe: ModuleType, seed_path: Path, prompt_version: str) -> None:
    config = recipe.build_config(seed_path=str(seed_path), prompt_version=prompt_version).build()
    columns = {column.name: column for column in config.columns}
    models = {model.alias: model for model in config.model_configs}

    assert list(columns) == ["question_type", "question", "answer", "quality_check"]
    assert columns["question_type"].params.values == list(recipe.QUESTION_TYPES)
    assert columns["question_type"].params.weights == list(recipe.QUESTION_TYPE_WEIGHTS)
    assert columns["question"].model_alias == "chart-vlm-question"
    assert columns["answer"].model_alias == "chart-vlm"
    assert columns["quality_check"].model_alias == "chart-vlm-quality"
    assert columns["answer"].extract_reasoning_content is True
    assert columns["question"].multi_modal_context[0].column_name == "image_url"
    assert columns["question"].multi_modal_context[0].data_type == "url"
    assert columns["question"].multi_modal_context[0].image_format is None
    assert columns["answer"].multi_modal_context == columns["question"].multi_modal_context
    assert columns["quality_check"].multi_modal_context == columns["question"].multi_modal_context
    assert "{{ answer__reasoning_content }}" in columns["quality_check"].prompt
    assert config.seed_config.sampling_strategy == recipe.dd.SamplingStrategy.ORDERED
    assert list(models) == ["chart-vlm", "chart-vlm-question", "chart-vlm-quality"]

    for model in models.values():
        inference = model.inference_parameters
        assert model.model == recipe.DEFAULT_VLM_MODEL
        assert model.provider == recipe.VLLM_PROVIDER_NAME
        assert model.skip_health_check is False
        assert inference.max_tokens == 50000
        assert inference.timeout == 1200
        assert inference.temperature == 0.6
        assert inference.top_p == 0.95
        assert inference.max_parallel_requests == 20
        assert inference.extra_body == recipe.PRODUCTION_SAMPLING


@pytest.mark.parametrize(
    ("prompt_version", "raw_prompts"),
    [
        (
            "level1",
            ("LEVEL1_QUESTION_PROMPT", "LEVEL1_ANSWER_PROMPT", "LEVEL1_QUALITY_PROMPT"),
        ),
        (
            "level2",
            (
                "LEVEL2_QUESTION_PROMPT",
                "LEVEL2_ANSWER_PROMPT",
                "LEVEL2_QUALITY_PROMPT",
            ),
        ),
    ],
)
def test_prompt_pack_matches_runtime_whitespace_handling(
    recipe: ModuleType, prompt_version: str, raw_prompts: tuple[str, str, str]
) -> None:
    prompt_pack = recipe.get_prompt_pack(prompt_version)

    assert tuple(prompt_pack) == tuple(getattr(recipe, name).rstrip() for name in raw_prompts)
    assert all(not prompt.endswith(("\n", "\r", " ", "\t")) for prompt in prompt_pack)

    if prompt_version == "level2":
        assert "DIFFICULTY FLOOR (required)" in prompt_pack.question
        assert "NO-ATTEMPT HANDLING" in prompt_pack.answer


def test_unknown_prompt_version_is_rejected(recipe: ModuleType) -> None:
    with pytest.raises(ValueError, match="Unknown prompt version"):
        recipe.get_prompt_pack("level3")


def test_create_dataset_uses_production_provider_settings(
    recipe: ModuleType, seed_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_builder = recipe.build_config(seed_path=str(seed_path))
    dataset_results = MagicMock()
    data_designer = MagicMock()
    data_designer.create.return_value = dataset_results
    data_designer_class = MagicMock(return_value=data_designer)
    monkeypatch.setattr(recipe, "DataDesigner", data_designer_class)

    result = recipe.create_dataset(
        config_builder,
        num_records=7,
        vllm_endpoint="http://localhost:8000/v1",
        prompt_version="level1",
        artifact_path="artifacts",
    )

    assert result is dataset_results
    provider = data_designer_class.call_args.kwargs["model_providers"][0]
    assert provider.name == "vllm-vl"
    assert provider.endpoint == "http://localhost:8000/v1"
    assert provider.provider_type == "openai"
    assert provider.api_key is None
    data_designer.create.assert_called_once_with(
        config_builder,
        num_records=7,
        dataset_name="chart_qa_level1",
    )
