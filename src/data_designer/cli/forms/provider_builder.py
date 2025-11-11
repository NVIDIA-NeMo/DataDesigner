# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from data_designer.cli.forms.builder import FormBuilder
from data_designer.cli.forms.field import SelectField, TextField
from data_designer.cli.forms.form import Form
from data_designer.cli.utils import validate_url
from data_designer.engine.model_provider import ModelProvider

# Predefined provider templates
PREDEFINED_PROVIDERS = {
    "nvidia": {
        "name": "nvidia",
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "provider_type": "openai",
        "api_key": "NVIDIA_API_KEY",
    },
    "openai": {
        "name": "openai",
        "endpoint": "https://api.openai.com/v1",
        "provider_type": "openai",
        "api_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "name": "anthropic",
        "endpoint": "https://api.anthropic.com/v1",
        "provider_type": "openai",
        "api_key": "ANTHROPIC_API_KEY",
    },
}


class ProviderFormBuilder(FormBuilder[ModelProvider]):
    """Builds interactive forms for provider configuration."""

    def __init__(self, existing_names: set[str] | None = None):
        super().__init__("Provider Configuration")
        self.existing_names = existing_names or set()

    def create_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the provider configuration form.

        For new providers without initial data, returns config_type selection.
        For updates (with initial_data), returns manual configuration form.
        """
        # For new providers (no initial_data), offer predefined options first
        if not initial_data:
            # Filter out already-used predefined providers
            available_predefined = {
                k: f"{v['name']} - {v['endpoint']}"
                for k, v in PREDEFINED_PROVIDERS.items()
                if v["name"] not in self.existing_names
            }

            if available_predefined:
                # Only show the config type selection
                fields = [
                    SelectField(
                        "config_type",
                        "Configuration type",
                        options={
                            "predefined": "Use a predefined provider",
                            "manual": "Configure one manually",
                        },
                        default="predefined",
                    )
                ]
                return Form(self.title, fields)

        # Manual configuration form (for updates or when predefined not available)
        return self._create_manual_form(initial_data)

    def _create_predefined_form(self) -> Form:
        """Create form for selecting a predefined provider."""
        available_predefined = {
            k: f"{v['name']} - {v['endpoint']}"
            for k, v in PREDEFINED_PROVIDERS.items()
            if v["name"] not in self.existing_names
        }

        fields = [
            SelectField(
                "predefined_choice",
                "Select predefined provider",
                options=available_predefined,
                default=list(available_predefined.keys())[0] if available_predefined else None,
            )
        ]

        return Form(self.title, fields)

    def _create_manual_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create form for manual provider configuration."""
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
            SelectField(
                "provider_type",
                "Provider type",
                options={"openai": "OpenAI-compatible API"},
                default=initial_data.get("provider_type", "openai") if initial_data else "openai",
            ),
            TextField(
                "api_key",
                "API key or environment variable name",
                default=initial_data.get("api_key") if initial_data else None,
                required=False,
            ),
        ]

        return Form(self.title, fields)

    def _create_api_key_only_form(self, initial_data: dict[str, Any]) -> Form:
        """Create form for updating only the API key (for predefined providers)."""
        fields = [
            TextField(
                "api_key",
                "API key or environment variable name",
                default=initial_data.get("api_key"),
                required=False,
            ),
        ]

        return Form(self.title, fields)

    def _is_predefined_provider(self, provider_name: str) -> bool:
        """Check if a provider is a predefined one."""
        return provider_name in PREDEFINED_PROVIDERS

    def run(self, initial_data: dict[str, Any] | None = None) -> ModelProvider | None:
        """Run the interactive form with conditional flow based on user choice."""
        from data_designer.cli.ui import confirm_action, console, print_error, print_info

        # For updates (initial_data provided)
        if initial_data:
            provider_name = initial_data.get("name")

            # For predefined providers, only allow updating API key
            if provider_name and self._is_predefined_provider(provider_name):
                print_info(f"Updating predefined provider '{provider_name}' (only API key can be changed)")
                console.print()

                api_key_form = self._create_api_key_only_form(initial_data)

                while True:
                    result = api_key_form.prompt_all(allow_back=False)

                    if result is None:
                        if confirm_action("Cancel configuration?", default=False):
                            return None
                        continue

                    try:
                        # Build config keeping all original fields except api_key
                        config = ModelProvider(
                            name=initial_data["name"],
                            endpoint=initial_data["endpoint"],
                            provider_type=initial_data["provider_type"],
                            api_key=result.get("api_key") or initial_data.get("api_key"),
                        )
                        return config
                    except Exception as e:
                        print_error(f"Configuration error: {e}")
                        if not confirm_action("Try again?", default=True):
                            return None
                        continue

            # For custom providers, allow updating all fields via parent class
            return super().run(initial_data)

        # For new providers, print message and handle conditional flow
        print_info(f"Starting {self.title}")
        console.print()

        # Check if any predefined providers are available
        available_predefined = {k: v for k, v in PREDEFINED_PROVIDERS.items() if v["name"] not in self.existing_names}

        # If no predefined providers available, go straight to manual form
        if not available_predefined:
            manual_form = self._create_manual_form()
            while True:
                manual_result = manual_form.prompt_all(allow_back=False)

                if manual_result is None:
                    if confirm_action("Cancel configuration?", default=False):
                        return None
                    continue

                try:
                    config = self.build_config(manual_result)
                    return config
                except Exception as e:
                    print_error(f"Configuration error: {e}")
                    if not confirm_action("Try again?", default=True):
                        return None
                    continue

        # For new providers with available predefined options, ask for config type
        type_form = self.create_form(initial_data=None)

        while True:
            type_result = type_form.prompt_all(allow_back=False)

            if type_result is None:
                if confirm_action("Cancel configuration?", default=False):
                    return None
                continue

            config_type = type_result.get("config_type", "manual")

            # Now show the appropriate form based on choice
            if config_type == "predefined":
                # Show predefined provider selection
                predefined_form = self._create_predefined_form()
                predefined_result = predefined_form.prompt_all(allow_back=True)

                if predefined_result is None:
                    # User cancelled, go back to type selection
                    continue

                try:
                    # Build config from predefined choice
                    predefined_key = predefined_result.get("predefined_choice")
                    if predefined_key and predefined_key in PREDEFINED_PROVIDERS:
                        predefined = PREDEFINED_PROVIDERS[predefined_key]

                        # Ask for API key with predefined default
                        api_key_form = self._create_api_key_only_form(predefined)
                        api_key_result = api_key_form.prompt_all(allow_back=True)

                        if api_key_result is None:
                            # User cancelled, go back to type selection
                            continue

                        # Use provided API key or fall back to predefined default
                        api_key = api_key_result.get("api_key") or predefined.get("api_key")

                        config = ModelProvider(
                            name=predefined["name"],
                            endpoint=predefined["endpoint"],
                            provider_type=predefined["provider_type"],
                            api_key=api_key,
                        )
                        return config
                    else:
                        print_error("Invalid predefined provider selection")
                        continue
                except Exception as e:
                    print_error(f"Configuration error: {e}")
                    if not confirm_action("Try again?", default=True):
                        return None
                    continue
            else:
                # Show manual configuration form
                manual_form = self._create_manual_form()
                manual_result = manual_form.prompt_all(allow_back=True)

                if manual_result is None:
                    # User cancelled, go back to type selection
                    continue

                try:
                    config = self.build_config(manual_result)
                    return config
                except Exception as e:
                    print_error(f"Configuration error: {e}")
                    if not confirm_action("Try again?", default=True):
                        return None
                    continue

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
        """Build ModelProvider from form data (manual configuration only)."""
        return ModelProvider(
            name=form_data["name"],
            endpoint=form_data["endpoint"],
            provider_type=form_data["provider_type"],
            api_key=form_data.get("api_key"),
        )
