# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from data_designer.cli.forms.builder import FormBuilder
from data_designer.cli.forms.field import NumericField, SelectField, TextField
from data_designer.cli.forms.form import Form
from data_designer.config.models import ModelConfig
from data_designer.config.utils.constants import MAX_TEMPERATURE, MAX_TOP_P, MIN_TEMPERATURE, MIN_TOP_P


class ModelFormBuilder(FormBuilder[ModelConfig]):
    """Builds interactive forms for model configuration."""

    def __init__(self, existing_aliases: set[str] | None = None, available_providers: list[str] | None = None):
        super().__init__("Model Configuration")
        self.existing_aliases = existing_aliases or set()
        self.available_providers = available_providers or []

    def create_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the model configuration form."""
        fields = []

        # Model alias
        fields.append(
            TextField(
                "alias",
                "Model alias (used in your configs)",
                default=initial_data.get("alias") if initial_data else None,
                required=True,
                validator=self._validate_alias,
            )
        )

        # Model ID
        fields.append(
            TextField(
                "model",
                "Model ID",
                default=initial_data.get("model") if initial_data else None,
                required=True,
                validator=lambda x: (False, "Model ID is required") if not x else (True, None),
            )
        )

        # Provider (if multiple available)
        if len(self.available_providers) > 1:
            provider_options = {p: p for p in self.available_providers}
            fields.append(
                SelectField(
                    "provider",
                    "Select provider for this model",
                    options=provider_options,
                    default=initial_data.get("provider", self.available_providers[0])
                    if initial_data
                    else self.available_providers[0],
                )
            )
        elif len(self.available_providers) == 1:
            # Single provider - will be set automatically
            pass

        # Inference parameters
        fields.extend(
            [
                NumericField(
                    "temperature",
                    f"Temperature ({MIN_TEMPERATURE}-{MAX_TEMPERATURE})",
                    default=initial_data.get("inference_parameters", {}).get("temperature", 0.7)
                    if initial_data
                    else 0.7,
                    min_value=MIN_TEMPERATURE,
                    max_value=MAX_TEMPERATURE,
                ),
                NumericField(
                    "top_p",
                    f"Top P ({MIN_TOP_P}-{MAX_TOP_P})",
                    default=initial_data.get("inference_parameters", {}).get("top_p", 0.9) if initial_data else 0.9,
                    min_value=MIN_TOP_P,
                    max_value=MAX_TOP_P,
                ),
                NumericField(
                    "max_tokens",
                    "Max tokens",
                    default=initial_data.get("inference_parameters", {}).get("max_tokens", 2048)
                    if initial_data
                    else 2048,
                    min_value=1,
                    max_value=100000,
                ),
            ]
        )

        return Form(self.title, fields)

    def _validate_alias(self, alias: str) -> tuple[bool, str | None]:
        """Validate model alias."""
        if not alias:
            return False, "Model alias is required"
        if alias in self.existing_aliases:
            return False, f"Model alias '{alias}' already exists"
        return True, None

    def build_config(self, form_data: dict[str, Any]) -> ModelConfig:
        """Build ModelConfig from form data."""
        # Determine provider
        if "provider" in form_data:
            provider = form_data["provider"]
        elif len(self.available_providers) == 1:
            provider = self.available_providers[0]
        else:
            provider = None

        return ModelConfig(
            alias=form_data["alias"],
            model=form_data["model"],
            provider=provider,
            inference_parameters={
                "temperature": form_data["temperature"],
                "top_p": form_data["top_p"],
                "max_tokens": int(form_data["max_tokens"]),
                "max_parallel_requests": 4,
            },
        )
