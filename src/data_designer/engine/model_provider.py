# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import Self

from pydantic import BaseModel, field_validator, model_validator

from data_designer.config.models import ModelProvider
from data_designer.engine.errors import NoModelProvidersError, UnknownProviderError


class ModelProviderRegistry(BaseModel):
    providers: list[ModelProvider]
    default: str | None = None

    @field_validator("providers", mode="after")
    @classmethod
    def validate_providers_not_empty(cls, v: list[ModelProvider]) -> list[ModelProvider]:
        if len(v) == 0:
            raise ValueError("At least one model provider must be defined")
        return v

    @field_validator("providers", mode="after")
    @classmethod
    def validate_providers_have_unique_names(cls, v: list[ModelProvider]) -> list[ModelProvider]:
        names = set()
        dupes = set()
        for provider in v:
            if provider.name in names:
                dupes.add(provider.name)
            names.add(provider.name)

        if len(dupes) > 0:
            raise ValueError(f"Model providers must have unique names, found duplicates: {dupes}")
        return v

    @model_validator(mode="after")
    def check_implicit_default(self) -> Self:
        if self.default is None and len(self.providers) != 1:
            raise ValueError("A default provider must be specified if multiple model providers are defined")
        return self

    @model_validator(mode="after")
    def check_default_exists(self) -> Self:
        if self.default and self.default not in self._providers_dict:
            raise ValueError(f"Specified default {self.default!r} not found in providers list")
        return self

    def get_default_provider_name(self) -> str:
        return self.default or self.providers[0].name

    @cached_property
    def _providers_dict(self) -> dict[str, ModelProvider]:
        return {p.name: p for p in self.providers}

    def get_provider(self, name: str | None) -> ModelProvider:
        if name is None:
            name = self.get_default_provider_name()

        try:
            return self._providers_dict[name]
        except KeyError:
            raise UnknownProviderError(f"No provider named {name!r} registered")


def resolve_model_provider_registry(
    model_providers: list[ModelProvider], default_provider_name: str | None = None
) -> ModelProviderRegistry:
    if len(model_providers) == 0:
        raise NoModelProvidersError("At least one model provider must be defined")
    return ModelProviderRegistry(
        providers=model_providers,
        default=default_provider_name or model_providers[0].name,
    )
