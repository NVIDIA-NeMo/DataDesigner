# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from data_designer.cli.controllers.agent_controller import AgentController
from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider


def test_get_model_aliases_state_reports_provider_status(tmp_path: Path) -> None:
    provider_repository = ProviderRepository(tmp_path)
    provider_repository.save(
        ModelProviderRegistry(
            providers=[
                ModelProvider(
                    name="provider-a",
                    endpoint="https://api.example.com/a",
                    provider_type="openai",
                    api_key="test-api-key",
                ),
                ModelProvider(
                    name="provider-b",
                    endpoint="https://api.example.com/b",
                    provider_type="openai",
                    api_key="MISSING_PROVIDER_KEY",
                ),
            ],
            default="provider-a",
        )
    )

    model_repository = ModelRepository(tmp_path)
    model_repository.save(
        ModelConfigRegistry(
            model_configs=[
                ModelConfig(
                    alias="alpha",
                    model="model-alpha",
                    provider=None,
                    inference_parameters=ChatCompletionInferenceParams(),
                ),
                ModelConfig(
                    alias="beta",
                    model="model-beta",
                    provider="provider-b",
                    inference_parameters=ChatCompletionInferenceParams(),
                ),
                ModelConfig(
                    alias="gamma",
                    model="model-gamma",
                    provider="provider-missing",
                    inference_parameters=ChatCompletionInferenceParams(),
                ),
            ]
        )
    )

    controller = AgentController(tmp_path)
    payload = controller.get_model_aliases_state()

    assert payload["model_config_present"] is True
    assert payload["provider_config_present"] is True
    assert payload["default_provider"] == "provider-a"
    assert payload["items"] == [
        {
            "model_alias": "alpha",
            "model": "model-alpha",
            "generation_type": "chat-completion",
            "configured_provider": None,
            "effective_provider": "provider-a",
            "usable": True,
            "reason": None,
        },
        {
            "model_alias": "beta",
            "model": "model-beta",
            "generation_type": "chat-completion",
            "configured_provider": "provider-b",
            "effective_provider": "provider-b",
            "usable": False,
            "reason": "Provider 'provider-b' is missing an API key.",
        },
        {
            "model_alias": "gamma",
            "model": "model-gamma",
            "generation_type": "chat-completion",
            "configured_provider": "provider-missing",
            "effective_provider": "provider-missing",
            "usable": False,
            "reason": "Provider 'provider-missing' is not configured.",
        },
    ]


def test_get_model_aliases_state_handles_missing_local_files(tmp_path: Path) -> None:
    controller = AgentController(tmp_path)

    payload = controller.get_model_aliases_state()

    assert payload == {
        "model_config_present": False,
        "provider_config_present": False,
        "default_provider": None,
        "items": [],
    }


def test_get_persona_datasets_state_reports_installed_locales(tmp_path: Path) -> None:
    managed_assets_dir = tmp_path / "managed-assets" / "datasets"
    managed_assets_dir.mkdir(parents=True)
    (managed_assets_dir / "en_US.parquet").write_text("stub")

    controller = AgentController(tmp_path)
    payload = controller.get_persona_datasets_state()

    assert payload["managed_assets_directory"] == str(managed_assets_dir)
    installed_by_locale = {item["locale"]: item["installed"] for item in payload["items"]}
    assert installed_by_locale["en_US"] is True
    assert any(not item["installed"] for item in payload["items"] if item["locale"] != "en_US")


def test_get_context_returns_self_describing_payload(tmp_path: Path) -> None:
    controller = AgentController(tmp_path)

    payload = controller.get_context()

    operation_names = [operation["name"] for operation in payload["operations"]]
    assert operation_names == [
        "context",
        "types",
        "schema",
        "builder",
        "state.model-aliases",
        "state.persona-datasets",
    ]
    assert payload["families"]
    assert "columns" in payload["types"]
    assert payload["builder"]["methods"]
    assert all(method["docstring"] is None for method in payload["builder"]["methods"])
