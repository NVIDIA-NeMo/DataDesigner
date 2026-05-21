# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from data_designer.engine.models.request_admission.resources import RequestDomain, RequestResourceKey
from data_designer.engine.models.resources import ProviderModelKey


@dataclass(frozen=True)
class ResolvedRequestResource:
    provider_model: ProviderModelKey
    resource: RequestResourceKey
    aliases: tuple[str, ...] = ()
    generation_kind: str | None = None


class RequestResourceResolver:
    """Canonical provider/model/domain request-resource identity factory."""

    def resolve(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: RequestDomain,
        model_alias: str | None = None,
        provider_alias: str | None = None,
        generation_kind: str | None = None,
    ) -> ResolvedRequestResource:
        resource = RequestResourceKey(provider_name=provider_name, model_id=model_id, domain=domain)
        aliases = tuple(alias for alias in (provider_alias, model_alias) if alias)
        return ResolvedRequestResource(
            provider_model=resource.provider_model_key,
            resource=resource,
            aliases=aliases,
            generation_kind=generation_kind,
        )
