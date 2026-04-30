# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic content-addressable fingerprint for a workflow config.

The fingerprint identifies the *data-relevant* portion of a
:class:`DataDesignerConfig` so that two configs producing the same dataset hash
to the same value, while configs differing only in environment, runtime, or
post-generation analysis hash to different values when they should and to the
same value when they shouldn't.

The hash is computed over a canonical JSON dump of the config (Pydantic
``model_dump(mode="json")``) with non-identity fields removed. Dict keys are
sorted, list order is preserved (list order is part of identity).

The normalization scheme is versioned via :data:`CONFIG_HASH_VERSION`. Persist
the version alongside the hash so future scheme changes can be detected as
"unknown identity" rather than "definite mismatch".
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from data_designer.config.column_configs import CustomColumnConfig

if TYPE_CHECKING:
    from data_designer.config.data_designer_config import DataDesignerConfig

logger = logging.getLogger(__name__)

CONFIG_HASH_VERSION = 1
CONFIG_HASH_ALGO = "sha256"

# Top-level DataDesignerConfig keys excluded from the fingerprint.
# tool_configs   -- MCP tool wiring is a runtime/execution choice
# profilers      -- post-generation analysis, doesn't affect generated rows
_EXCLUDED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"tool_configs", "profilers"})

# ModelConfig keys excluded -- env/runtime knobs.
_EXCLUDED_MODEL_KEYS: frozenset[str] = frozenset({"skip_health_check"})

# Inference-parameter keys excluded -- concurrency / timing only.
_EXCLUDED_INFERENCE_KEYS: frozenset[str] = frozenset({"max_parallel_requests", "timeout"})

# HuggingFaceSeedSource keys excluded -- auth and endpoint URL are not data identity.
_EXCLUDED_HF_SEED_KEYS: frozenset[str] = frozenset({"token", "endpoint"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fingerprint_config(
    config: DataDesignerConfig,
    *,
    custom_column_source: bool = False,
) -> dict[str, str | int]:
    """Compute a deterministic fingerprint of a workflow config.

    The fingerprint is content-addressable: identical configs (modulo excluded
    fields) produce identical hashes across processes, Python versions, and
    module load orders. Changing any identity-relevant field changes the hash;
    changing an excluded field does not.

    Identity-relevant fields:
      * ``columns`` (names, types, generator params, processors, validators,
        skip/drop/propagate_skip flags)
      * ``model_configs`` (alias, model, provider, sampling-relevant inference
        params -- temperature, top_p, max_tokens, extra_body)
      * ``seed_config`` (source path / sampling strategy / selection strategy)
      * ``constraints``
      * top-level ``processors``

    Excluded fields:
      * ``tool_configs`` (runtime tool wiring)
      * ``profilers`` (post-generation analysis)
      * ``model_configs[*].skip_health_check``
      * ``inference_parameters.max_parallel_requests``, ``inference_parameters.timeout``
      * HuggingFace seed source ``token`` and ``endpoint``

    Custom column generators are always identified by registered function name
    and ``generator_params`` (L1). When ``custom_column_source=True``, the
    function source is also hashed (L2); plugins whose source cannot be
    retrieved degrade gracefully with a warning.

    Note: ``buffer_size`` lives on :class:`RunConfig`, not on
    :class:`DataDesignerConfig`, and is therefore not part of this fingerprint.
    The fingerprint identifies the workflow definition; runtime knobs that
    don't change the final dataset are out of scope.

    Limitations:
      * **L1 collisions on ``__name__``**: custom columns are identified at L1
        by the generator's bare ``__name__``, not its qualified module path.
        Two unrelated generators in different modules with the same name and
        identical ``generator_params`` will produce the same L1 hash. Pass
        ``custom_column_source=True`` to disambiguate via source.
      * **L2 hashes raw source**: comment-only and formatting changes to a
        generator's source will change the L2 hash even though they don't
        affect behavior.

    Args:
        config: The workflow config to fingerprint.
        custom_column_source: If True, also hash ``inspect.getsource()`` of
            each custom column generator (L2). Defaults to False (L1 only).

    Returns:
        A dict with ``config_hash`` (``"sha256:..."``), ``config_hash_algo``,
        and ``config_hash_version`` suitable for embedding in dataset metadata.
    """
    payload: dict[str, Any] = {"config": _normalize_config_dict(config.to_dict())}
    if custom_column_source:
        payload["custom_column_sources"] = _collect_custom_column_sources(config)
    # No `default=` fallback: a non-JSON-native value would silently break determinism (e.g., repr with memory addresses).
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "config_hash": f"{CONFIG_HASH_ALGO}:{digest}",
        "config_hash_algo": CONFIG_HASH_ALGO,
        "config_hash_version": CONFIG_HASH_VERSION,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _drop_keys(source: dict[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in source.items() if k not in keys}


def _normalize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    normalized = _drop_keys(model_config, _EXCLUDED_MODEL_KEYS)
    inference_params = normalized.get("inference_parameters")
    if isinstance(inference_params, dict):
        normalized["inference_parameters"] = _drop_keys(inference_params, _EXCLUDED_INFERENCE_KEYS)
    return normalized


def _normalize_seed_config(seed_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(seed_config)
    seed_source = normalized.get("source")
    if isinstance(seed_source, dict) and seed_source.get("seed_type") == "hf":
        normalized["source"] = _drop_keys(seed_source, _EXCLUDED_HF_SEED_KEYS)
    return normalized


def _normalize_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = _drop_keys(config_dict, _EXCLUDED_TOP_LEVEL_KEYS)
    model_configs = normalized.get("model_configs")
    if model_configs:
        normalized["model_configs"] = [_normalize_model_config(mc) for mc in model_configs]
    seed_config = normalized.get("seed_config")
    if seed_config:
        normalized["seed_config"] = _normalize_seed_config(seed_config)
    return normalized


def _hash_custom_column_source(fn: Callable[..., Any], column_name: str) -> str | None:
    """Hash the source of a custom column generator (L2).

    Returns the sha256 hex digest of the function source, or ``None`` if the
    source cannot be retrieved (e.g., compiled / zipped plugin, C extension,
    interactively-defined function). Plugins that can't be source-hashed
    degrade gracefully with a warning rather than raising.
    """
    try:
        unwrapped = inspect.unwrap(fn)
        source = inspect.getsource(unwrapped)
    except (OSError, TypeError) as exc:
        logger.warning(
            "Could not retrieve source for custom column %r generator (%s); "
            "fingerprint will not detect implementation changes for this column.",
            column_name,
            exc,
        )
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _collect_custom_column_sources(config: DataDesignerConfig) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for col in config.columns:
        if isinstance(col, CustomColumnConfig):
            sources.append(
                {
                    "name": col.name,
                    "source_hash": _hash_custom_column_source(col.generator_function, col.name),
                }
            )
    return sources
