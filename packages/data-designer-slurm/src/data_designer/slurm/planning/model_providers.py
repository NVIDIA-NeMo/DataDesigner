# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve native DD provider settings into non-secret, immutable run inputs."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from urllib.parse import urlsplit

from pydantic import JsonValue

from data_designer.config.models import ModelProvider
from data_designer.slurm.config.environment import is_secret_bearing_name, validate_no_plaintext_secrets
from data_designer.slurm.contracts import validate_plain_text, validate_url
from data_designer.slurm.planning.builder_identity import get_declared_model_aliases


def validate_hosted_model_providers(providers: Sequence[ModelProvider]) -> tuple[ModelProvider, ...]:
    """Preserve DD settings while admitting only environment-backed credentials."""
    names = [provider.name for provider in providers]
    if len(names) != len(set(names)):
        raise ValueError("hosted model provider names must be unique")
    for provider in providers:
        if not provider.name:
            raise ValueError("hosted model providers require a name")
        validate_url(provider.endpoint, field_name="hosted model provider endpoint")
        parsed = urlsplit(provider.endpoint)
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("hosted model provider endpoints must not contain credentials, queries, or fragments")
        if provider.api_key is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", provider.api_key) is None:
            raise ValueError("hosted model provider API keys must reference environment variables")
        validate_no_plaintext_secrets(provider.extra_body, field_name="hosted model provider extra_body")
        for name, value in (provider.extra_headers or {}).items():
            if re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name) is None:
                raise ValueError("hosted model provider headers require valid HTTP field names")
            if is_secret_bearing_name(name) or name.casefold() in {"authorization", "proxy-authorization", "cookie"}:
                raise ValueError("hosted model providers do not support sensitive headers")
            validate_plain_text(value, field_name="hosted model provider header")
    return tuple(
        ModelProvider.model_validate_json(json.dumps(provider.model_dump(mode="json"), allow_nan=False))
        for provider in providers
    )


def resolve_hosted_model_providers(
    builder: dict[str, JsonValue],
    local_aliases: Collection[str],
    providers: Sequence[ModelProvider],
) -> tuple[ModelProvider, ...]:
    """Snapshot only the native providers required by aliases not deployed locally."""
    get_declared_model_aliases(builder)
    envelope = builder.get("data_designer", builder)
    assert isinstance(envelope, dict)
    configs = envelope.get("model_configs") or []
    assert isinstance(configs, list)
    required: list[str] = []
    for config in configs:
        assert isinstance(config, dict)
        if config["alias"] in local_aliases:
            continue
        name = config.get("provider")
        if not isinstance(name, str) or not name:
            raise ValueError("every hosted model must reference a native Data Designer provider")
        if name not in required:
            required.append(name)
    captured: list[ModelProvider] = []
    for name in required:
        matches = [provider for provider in providers if provider.name == name]
        if len(matches) != 1:
            raise ValueError("each hosted model must resolve to exactly one Data Designer provider")
        if name in {f"slurm-{alias}" for alias in local_aliases}:
            raise ValueError("hosted provider names must not collide with managed providers")
        captured.append(matches[0])
    return validate_hosted_model_providers(captured)


def load_hosted_model_providers(snapshots: Sequence[Mapping[str, JsonValue]]) -> tuple[ModelProvider, ...]:
    """Restore DD's provider objects from their captured native JSON representation."""
    return validate_hosted_model_providers(
        tuple(ModelProvider.model_validate_json(json.dumps(value, allow_nan=False)) for value in snapshots)
    )


def dump_hosted_model_providers(providers: Sequence[ModelProvider]) -> tuple[dict[str, JsonValue], ...]:
    """Serialize native settings for freezing by the Slurm run contract."""
    return tuple(provider.model_dump(mode="json") for provider in validate_hosted_model_providers(providers))
