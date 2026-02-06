# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health checks for all predefined model providers.

Verifies that each model in each provider can respond to a basic request.
Providers without an API key set in the environment are skipped.

Usage:
    uv run python scripts/health_checks.py
"""

import os
import sys
import traceback

from data_designer.config.models import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    ModelConfig,
    ModelProvider,
)
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
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.secret_resolver import EnvironmentResolver

PROVIDER_API_KEY_ENV_VARS = {
    NVIDIA_PROVIDER_NAME: NVIDIA_API_KEY_ENV_VAR_NAME,
    OPENAI_PROVIDER_NAME: OPENAI_API_KEY_ENV_VAR_NAME,
    OPENROUTER_PROVIDER_NAME: OPENROUTER_API_KEY_ENV_VAR_NAME,
}


def _get_provider_registry(provider_name: str) -> ModelProviderRegistry:
    provider_data = next(p for p in PREDEFINED_PROVIDERS if p["name"] == provider_name)
    provider = ModelProvider(**provider_data)
    return ModelProviderRegistry(providers=[provider])


def _check_model(provider_name: str, model_type: str) -> None:
    provider_registry = _get_provider_registry(provider_name)
    secret_resolver = EnvironmentResolver()

    model_info = PREDEFINED_PROVIDERS_MODEL_MAP[provider_name][model_type]
    model_name = model_info["model"]
    inference_params = model_info["inference_parameters"]

    if model_type == "embedding":
        params = EmbeddingInferenceParams(**inference_params)
    else:
        params = ChatCompletionInferenceParams(**inference_params)

    model_config = ModelConfig(
        alias=f"{provider_name}-{model_type}",
        model=model_name,
        inference_parameters=params,
        provider=provider_name,
    )

    facade = ModelFacade(model_config, secret_resolver, provider_registry)

    if model_type == "embedding":
        result = facade.generate_text_embeddings(
            input_texts=["Hello!"],
            skip_usage_tracking=True,
        )
        assert len(result) == 1 and len(result[0]) > 0
    else:
        result, _ = facade.generate(
            prompt="Say 'OK' and nothing else.",
            parser=lambda x: x,
            system_prompt="You are a helpful assistant.",
            max_correction_steps=0,
            max_conversation_restarts=0,
            skip_usage_tracking=True,
        )
        assert isinstance(result, str) and len(result) > 0


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
