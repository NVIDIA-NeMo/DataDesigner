# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from data_designer.cli.forms.builder import FormBuilder
from data_designer.cli.forms.field import TextField
from data_designer.cli.forms.form import Form
from data_designer.cli.utils import validate_url
from data_designer.engine.model_provider import ModelProvider


class ProviderFormBuilder(FormBuilder[ModelProvider]):
    """Builds interactive forms for provider configuration."""

    def __init__(self, existing_names: set[str] | None = None):
        super().__init__("Provider Configuration")
        self.existing_names = existing_names or set()

    def create_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the provider configuration form."""
        fields = [
            TextField(
                "name",
                "Provider name",
                default=initial_data.get("name") if initial_data else None,
                required=True,
                validator=self._validate_name,
            ),
            TextField(
                "endpoint",
                "API endpoint URL",
                default=initial_data.get("endpoint") if initial_data else None,
                required=True,
                validator=self._validate_endpoint,
            ),
            TextField(
                "provider_type",
                "Provider type",
                default=initial_data.get("provider_type", "openai") if initial_data else "openai",
                required=True,
            ),
            TextField(
                "api_key",
                "API key or environment variable name",
                default=initial_data.get("api_key") if initial_data else None,
                required=False,
            ),
        ]

        return Form(self.title, fields)

    def _validate_name(self, name: str) -> tuple[bool, str | None]:
        """Validate provider name."""
        if not name:
            return False, "Provider name is required"
        if name in self.existing_names:
            return False, f"Provider '{name}' already exists"
        return True, None

    def _validate_endpoint(self, endpoint: str) -> tuple[bool, str | None]:
        """Validate endpoint URL."""
        if not endpoint:
            return False, "Endpoint URL is required"
        if not validate_url(endpoint):
            return False, "Invalid URL format (must start with http:// or https://)"
        return True, None

    def build_config(self, form_data: dict[str, Any]) -> ModelProvider:
        """Build ModelProvider from form data."""
        return ModelProvider(
            name=form_data["name"],
            endpoint=form_data["endpoint"],
            provider_type=form_data["provider_type"],
            api_key=form_data.get("api_key"),
        )
