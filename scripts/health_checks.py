# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health checks for all predefined model providers.

Verifies that each model in each provider can respond to a basic request.
Providers without an API key set in the environment are skipped.

Usage:
    uv run python scripts/health_checks.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import data_designer.config as dd
from data_designer.config.utils.constants import (
    NVIDIA_API_KEY_ENV_VAR_NAME,
    NVIDIA_PROVIDER_NAME,
    OPENAI_API_KEY_ENV_VAR_NAME,
    OPENAI_PROVIDER_NAME,
    OPENROUTER_API_KEY_ENV_VAR_NAME,
    OPENROUTER_PROVIDER_NAME,
    PREDEFINED_PROVIDERS,
    PREDEFINED_PROVIDERS_MODEL_MAP,
)
from data_designer.interface import DataDesigner

PROVIDER_API_KEY_ENV_VARS = {
    NVIDIA_PROVIDER_NAME: NVIDIA_API_KEY_ENV_VAR_NAME,
    OPENAI_PROVIDER_NAME: OPENAI_API_KEY_ENV_VAR_NAME,
    OPENROUTER_PROVIDER_NAME: OPENROUTER_API_KEY_ENV_VAR_NAME,
}


def _get_provider(provider_name: str) -> dd.ModelProvider:
    provider_data = next(p for p in PREDEFINED_PROVIDERS if p["name"] == provider_name)
    return dd.ModelProvider(**provider_data)


def _get_model_config(provider_name: str, model_type: str) -> dd.ModelConfig:
    model_info = PREDEFINED_PROVIDERS_MODEL_MAP[provider_name][model_type]
    model_name = model_info["model"]
    inference_params = model_info["inference_parameters"]

    if model_type == "embedding":
        params = dd.EmbeddingInferenceParams(**inference_params)
    else:
        params = dd.ChatCompletionInferenceParams(**inference_params)

    return dd.ModelConfig(
        alias=f"{provider_name}-{model_type}",
        model=model_name,
        inference_parameters=params,
        provider=provider_name,
    )


def _build_check_config(model_config: dd.ModelConfig, model_type: str) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    if model_type == "embedding":
        builder.add_column(
            dd.SamplerColumnConfig(
                name="input_text",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=["Hello!"]),
            )
        )
        builder.add_column(
            dd.EmbeddingColumnConfig(
                name="embedding",
                target_column="input_text",
                model_alias=model_config.alias,
            )
        )
    else:
        builder.add_column(dd.LLMTextColumnConfig(name="response", prompt="Hello!", model_alias=model_config.alias))
    return builder


def _check_model(provider_name: str, model_type: str) -> None:
    provider = _get_provider(provider_name)
    model_config = _get_model_config(provider_name, model_type)
    config_builder = _build_check_config(model_config, model_type)

    with TemporaryDirectory(prefix="data-designer-health-check-") as temp_dir:
        DataDesigner(artifact_path=Path(temp_dir), model_providers=[provider]).check_models(config_builder)


def main() -> int:
    passed, failed, skipped = 0, 0, 0

    for provider_name, env_var in PROVIDER_API_KEY_ENV_VARS.items():
        if not os.environ.get(env_var):
            models = list(PREDEFINED_PROVIDERS_MODEL_MAP[provider_name])
            skipped += len(models)
            print(f"SKIP  {provider_name} ({env_var} not set)")
            continue

        for model_type in PREDEFINED_PROVIDERS_MODEL_MAP[provider_name]:
            label = f"{provider_name}/{model_type}"
            try:
                _check_model(provider_name, model_type)
                passed += 1
                print(f"PASS  {label}")
            except Exception:
                failed += 1
                tb = traceback.format_exc()
                print(f"FAIL  {label}\n{tb}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
