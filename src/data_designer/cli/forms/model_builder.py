# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.cli.forms.builder import FormBuilder
from data_designer.cli.forms.field import NumericField, SelectField, TextField
from data_designer.cli.forms.form import Form
from data_designer.cli.ui import confirm_action, print_error, print_text
from data_designer.config.models import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    GenerationType,
    ModelConfig,
)


class ModelFormBuilder(FormBuilder[ModelConfig]):
    """Builds interactive forms for model configuration."""

    def __init__(self, existing_aliases: set[str] | None = None, available_providers: list[str] | None = None):
        super().__init__("Model Configuration")
        self.existing_aliases = existing_aliases or set()
        self.available_providers = available_providers or []

    def create_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the model configuration form with basic fields."""
        fields = []

        # Model alias
        fields.append(
            TextField(
                "alias",
                "Model alias (used in your configs)",
                default=initial_data.get("alias") if initial_data else None,
                required=True,
                validator=self.validate_alias,
            )
        )

        # Model ID
        fields.append(
            TextField(
                "model",
                "Model",
                default=initial_data.get("model") if initial_data else None,
                required=True,
                validator=lambda x: (False, "Model is required") if not x else (True, None),
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

        # Generation type
        # Extract from inference_parameters if present (for existing models)
        default_gen_type = GenerationType.CHAT_COMPLETION
        if initial_data:
            inference_params = initial_data.get("inference_parameters", {})
            default_gen_type = inference_params.get("generation_type", default_gen_type)

        fields.append(
            SelectField(
                "generation_type",
                "Generation type",
                options={
                    GenerationType.CHAT_COMPLETION: "Chat completion",
                    GenerationType.EMBEDDING: "Embedding",
                },
                default=default_gen_type,
            )
        )

        return Form(self.title, fields)

    def create_inference_params_form(
        self, generation_type: GenerationType, initial_params: dict[str, Any] | None = None
    ) -> Form:
        """Create generation-type-specific inference parameters form."""
        initial_params = initial_params or {}
        fields = []

        if generation_type == GenerationType.CHAT_COMPLETION:
            # Temperature
            fields.append(
                NumericField(
                    "temperature",
                    "Temperature <dim>(0.0-2.0)</dim>",
                    default=initial_params.get("temperature"),
                    min_value=0.0,
                    max_value=2.0,
                    required=False,
                    help_text="Higher values make output more random, lower values more deterministic",
                )
            )

            # Top P
            fields.append(
                NumericField(
                    "top_p",
                    "Top P <dim>(0.0-1.0)</dim>",
                    default=initial_params.get("top_p"),
                    min_value=0.0,
                    max_value=1.0,
                    required=False,
                    help_text="Controls diversity via nucleus sampling",
                )
            )

            # Max tokens
            fields.append(
                NumericField(
                    "max_tokens",
                    "Max tokens <dim>(maximum tokens to generate in response)</dim>",
                    default=initial_params.get("max_tokens"),
                    min_value=1.0,
                    required=False,
                    help_text="Maximum number of tokens to generate in the response",
                )
            )

            # Max parallel requests
            fields.append(
                NumericField(
                    "max_parallel_requests",
                    "Max parallel requests <dim>(default: 4)</dim>",
                    default=initial_params.get("max_parallel_requests", 4),
                    min_value=1.0,
                    required=False,
                    help_text="Maximum number of parallel API requests",
                )
            )

            # Timeout
            fields.append(
                NumericField(
                    "timeout",
                    "Timeout in seconds <dim>(optional)</dim>",
                    default=initial_params.get("timeout"),
                    min_value=1.0,
                    required=False,
                    help_text="Timeout for each API request in seconds",
                )
            )

        else:  # EMBEDDING
            # Encoding format
            fields.append(
                TextField(
                    "encoding_format",
                    "Encoding format <dim>(float or base64)</dim>",
                    default=initial_params.get("encoding_format"),
                    required=False,
                    validator=self.validate_encoding_format,
                )
            )

            # Dimensions
            fields.append(
                NumericField(
                    "dimensions",
                    "Dimensions <dim>(number of dimensions for embeddings)</dim>",
                    default=initial_params.get("dimensions"),
                    min_value=1.0,
                    required=False,
                    help_text="Model-specific dimension size (e.g., 1024, 768)",
                )
            )

            # Max parallel requests (common field)
            fields.append(
                NumericField(
                    "max_parallel_requests",
                    "Max parallel requests <dim>(default: 4)</dim>",
                    default=initial_params.get("max_parallel_requests", 4),
                    min_value=1.0,
                    required=False,
                    help_text="Maximum number of parallel API requests",
                )
            )

            # Timeout (common field)
            fields.append(
                NumericField(
                    "timeout",
                    "Timeout in seconds <dim>(optional)</dim>",
                    default=initial_params.get("timeout"),
                    min_value=1.0,
                    required=False,
                    help_text="Timeout for each API request in seconds",
                )
            )

        return Form(f"{self.title} - Inference Parameters", fields)

    def build_inference_params(self, generation_type: GenerationType, params_data: dict[str, Any]) -> dict[str, Any]:
        """Build inference parameters dictionary from form data with proper type conversions."""
        inference_params = {}

        if generation_type == GenerationType.CHAT_COMPLETION:
            if params_data.get("temperature") is not None:
                inference_params["temperature"] = params_data["temperature"]
            if params_data.get("top_p") is not None:
                inference_params["top_p"] = params_data["top_p"]
            if params_data.get("max_tokens") is not None:
                inference_params["max_tokens"] = int(params_data["max_tokens"])

        else:  # EMBEDDING
            # Only include fields with actual values; Pydantic will use defaults for missing fields
            if params_data.get("encoding_format"):
                inference_params["encoding_format"] = params_data["encoding_format"]
            if params_data.get("dimensions"):
                inference_params["dimensions"] = int(params_data["dimensions"])

        # Common fields for both generation types
        if params_data.get("max_parallel_requests") is not None:
            inference_params["max_parallel_requests"] = int(params_data["max_parallel_requests"])
        if params_data.get("timeout") is not None:
            inference_params["timeout"] = int(params_data["timeout"])

        return inference_params

    def run(self, initial_data: dict[str, Any] | None = None) -> ModelConfig | None:
        """Run the interactive form with two-step process for generation-type-specific parameters."""
        # Step 1: Collect basic model configuration
        basic_form = self.create_form(initial_data)

        if initial_data:
            basic_form.set_values(initial_data)

        while True:
            basic_result = basic_form.prompt_all(allow_back=True)

            if basic_result is None:
                if confirm_action("Cancel configuration?", default=False):
                    return None
                continue

            # Step 2: Collect generation-type-specific inference parameters
            generation_type = basic_result.get("generation_type", GenerationType.CHAT_COMPLETION)
            initial_params = initial_data.get("inference_parameters") if initial_data else None

            # Print message to indicate we're now configuring inference parameters
            gen_type_name = "chat completion" if generation_type == GenerationType.CHAT_COMPLETION else "embedding"
            print_text(
                f"⚙️  Configuring {gen_type_name} inference parameters [dim](Press Enter to keep current value or skip)[/dim]\n"
            )

            params_form = self.create_inference_params_form(generation_type, initial_params)

            params_result = params_form.prompt_all(allow_back=True)

            if params_result is None:
                if confirm_action("Cancel configuration?", default=False):
                    return None
                continue

            # Build inference_parameters dict from individual fields
            inference_params = self.build_inference_params(generation_type, params_result)

            # Merge results
            full_data = {**basic_result, "inference_parameters": inference_params}

            try:
                config = self.build_config(full_data)
                return config
            except Exception as e:
                print_error(f"Configuration error: {e}")
                if not confirm_action("Try again?", default=True):
                    return None

    def build_config(self, form_data: dict[str, Any]) -> ModelConfig:
        """Build ModelConfig from form data."""
        # Determine provider
        if "provider" in form_data:
            provider = form_data["provider"]
        elif len(self.available_providers) == 1:
            provider = self.available_providers[0]
        else:
            provider = None

        # Get generation type (from form data, used to determine which inference params to create)
        generation_type = form_data.get("generation_type", GenerationType.CHAT_COMPLETION)

        # Get inference parameters dict
        inference_params_dict = form_data.get("inference_parameters", {})

        # Create the appropriate inference parameters type based on generation_type
        # The generation_type will be set automatically by the inference params class
        if generation_type == GenerationType.EMBEDDING:
            inference_params = EmbeddingInferenceParams(**inference_params_dict)
        else:
            inference_params = ChatCompletionInferenceParams(**inference_params_dict)

        return ModelConfig(
            alias=form_data["alias"],
            model=form_data["model"],
            provider=provider,
            inference_parameters=inference_params,
        )

    def validate_alias(self, alias: str) -> tuple[bool, str | None]:
        """Validate model alias."""
        if not alias:
            return False, "Model alias is required"
        if alias in self.existing_aliases:
            return False, f"Model alias '{alias}' already exists"
        return True, None

    def validate_encoding_format(self, value: str) -> tuple[bool, str | None]:
        """Validate encoding format for embedding models."""
        if not value:
            return True, None  # Optional field
        if value.lower() in ("clear", "none", "default"):
            return True, None  # Allow clearing keywords
        if value not in ("float", "base64"):
            return False, "Must be either 'float' or 'base64'"
        return True, None
