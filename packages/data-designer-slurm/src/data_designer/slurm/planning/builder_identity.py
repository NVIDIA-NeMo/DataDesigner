# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submission-safe identity extraction for Data Designer builder payloads."""

from __future__ import annotations

from pydantic import JsonValue

from data_designer.slurm.contracts import ModelAlias, Sha256Digest, compute_serialized_json_sha256


def get_declared_model_aliases(builder: dict[str, JsonValue]) -> tuple[ModelAlias, ...]:
    """Return aliases declared by the known ``model_configs`` envelope."""
    data_designer = builder.get("data_designer", builder)
    if not isinstance(data_designer, dict):
        raise ValueError("builder data_designer value must be an object")
    model_configs = data_designer.get("model_configs") or []
    if not isinstance(model_configs, list):
        raise ValueError("builder model_configs must be a list")

    model_aliases: list[ModelAlias] = []
    for model_config in model_configs:
        if not isinstance(model_config, dict) or not isinstance(model_config.get("alias"), str):
            raise ValueError("each builder model config must contain a string alias")
        model_aliases.append(model_config["alias"])
    return tuple(model_aliases)


def get_persisted_builder_identity(
    builder: dict[str, JsonValue],
) -> tuple[tuple[ModelAlias, ...], Sha256Digest]:
    """Return declared aliases and the digest of persisted builder JSON."""
    return get_declared_model_aliases(builder), compute_serialized_json_sha256(builder)
