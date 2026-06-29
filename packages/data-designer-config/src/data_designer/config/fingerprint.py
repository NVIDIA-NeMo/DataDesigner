# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic content-addressable fingerprint for a workflow config.

The fingerprint identifies the *data-relevant* portion of a `DataDesignerConfig`
so that two configs producing the same dataset hash to the same value, while
configs differing only in environment, runtime, or post-generation analysis
hash to different values when they should and to the same value when they
shouldn't.

The hash is computed over a canonical JSON dump of the config (Pydantic
`model_dump(mode="json")`) with non-identity fields removed. Column order is
part of identity (DAG ordering); alias-keyed lookup tables (`model_configs`,
`tool_configs`) are sorted by alias so their internal order is irrelevant.
Empty/`None` optional collections are canonicalized to a single representation
so that builder-API and YAML-loaded configs producing identical datasets
fingerprint identically.

The normalization scheme is versioned via `CONFIG_HASH_VERSION`. Persist the
version alongside the hash so future scheme changes can be detected as
"unknown identity" rather than "definite mismatch".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from data_designer.config.column_configs import CustomColumnConfig

if TYPE_CHECKING:
    from data_designer.config.data_designer_config import DataDesignerConfig

CONFIG_HASH_VERSION = 2
CONFIG_HASH_ALGO = "sha256"


# ---------------------------------------------------------------------------
# Excluded fields (single canonical table). Each entry is excluded from the
# fingerprint because it doesn't affect generated rows:
#
#   profilers                          : post-generation analysis
#   model_configs[*].skip_health_check : startup probe, not generation
#   inference_parameters.{max_parallel_requests, timeout}
#                                      : concurrency / timing only
#   tool_configs[*].timeout_sec        : per-call timing knob
#   HuggingFaceSeedSource.{token, endpoint}
#                                      : auth + env, not data identity
# ---------------------------------------------------------------------------
_EXCLUDED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"profilers"})
_EXCLUDED_MODEL_KEYS: frozenset[str] = frozenset({"skip_health_check"})
_EXCLUDED_INFERENCE_KEYS: frozenset[str] = frozenset({"max_parallel_requests", "timeout"})
_EXCLUDED_TOOL_CONFIG_KEYS: frozenset[str] = frozenset({"timeout_sec"})
_EXCLUDED_HF_SEED_KEYS: frozenset[str] = frozenset({"token", "endpoint"})

# Optional collections whose `None` and `[]` representations must collapse so
# that builder-API and YAML-loaded configs producing identical datasets
# fingerprint identically.
_TOP_LEVEL_OPTIONAL_COLLECTIONS: frozenset[str] = frozenset(
    {"model_configs", "tool_configs", "constraints", "processors"}
)
_TOOL_CONFIG_OPTIONAL_COLLECTIONS: frozenset[str] = frozenset({"allow_tools"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fingerprint_config(config: DataDesignerConfig) -> dict[str, str | int]:
    """Compute a deterministic fingerprint of a workflow config.

    The fingerprint is content-addressable: identical configs (modulo excluded
    fields) produce identical hashes across processes, Python versions, and
    module load orders. Changing any identity-relevant field changes the hash;
    changing an excluded field does not.

    Identity-relevant fields:
      * `columns` - names, types, generator params, processors, validators,
        skip/drop flags. Column order is part of identity (DAG ordering).
      * `model_configs` - alias, model, provider, sampling-relevant inference
        params (temperature, top_p, max_tokens, extra_body). Sorted by alias.
      * `tool_configs` - alias, providers, allow_tools, max_tool_call_turns
        (the set of MCP tools shapes generation). Sorted by tool_alias.
      * `seed_config` - source path, sampling strategy, selection strategy.
      * `constraints`, top-level `processors`.

    See module-level constants for the canonical excluded-fields table.

    Custom column generators contribute their function's `__name__`,
    `__qualname__`, `__module__`, `generator_params`, and the decorator
    metadata set by `@custom_column_generator()` (`required_columns`,
    `side_effect_columns`, `model_aliases`).

    Limitation: closures captured via factory functions (e.g. `make_gen(factor)`
    returning a `gen` whose body references `factor`) share `__name__`,
    `__qualname__`, `__module__`, and source text, so two closures with
    different captured state will fingerprint identically. The fingerprint
    cannot see closure cell values.

    Args:
        config: The workflow config to fingerprint.

    Returns:
        A dict with `config_hash` (`"sha256:..."`), `config_hash_algo`, and
        `config_hash_version` suitable for embedding in dataset metadata.
    """
    payload = _normalize_config_dict(config.to_dict(), config)
    # No `default=` fallback: a non-JSON-native value would silently break determinism (e.g. repr with memory addresses).
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


def _drop_keys(source: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    keyset = set(keys)
    return {k: v for k, v in source.items() if k not in keyset}


def _drop_empty_optional(source: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Drop keys whose value is `None` or an empty list.

    `None` and `[]` are user-equivalent for optional collection fields; this
    collapses both to "absent" before hashing.
    """
    keyset = set(keys)
    return {k: v for k, v in source.items() if not (k in keyset and (v is None or v == []))}


def _normalize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    normalized = _drop_keys(model_config, _EXCLUDED_MODEL_KEYS)
    inference_params = normalized.get("inference_parameters")
    if isinstance(inference_params, dict):
        normalized["inference_parameters"] = _drop_keys(inference_params, _EXCLUDED_INFERENCE_KEYS)
    return normalized


def _normalize_tool_config(tool_config: dict[str, Any]) -> dict[str, Any]:
    normalized = _drop_keys(tool_config, _EXCLUDED_TOOL_CONFIG_KEYS)
    return _drop_empty_optional(normalized, _TOOL_CONFIG_OPTIONAL_COLLECTIONS)


def _normalize_seed_config(seed_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(seed_config)
    seed_source = normalized.get("source")
    if isinstance(seed_source, dict) and seed_source.get("seed_type") == "hf":
        normalized["source"] = _drop_keys(seed_source, _EXCLUDED_HF_SEED_KEYS)
    return normalized


def _enrich_custom_columns(config: DataDesignerConfig, columns_dump: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace each custom column's serialized `generator_function` (just the
    bare `__name__`) with a richer identity dict that includes `__qualname__`,
    `__module__`, and the `@custom_column_generator()` decorator metadata.

    Walks `config.columns` and `columns_dump` in lockstep so positional
    correspondence is reliable.
    """
    enriched: list[dict[str, Any]] = []
    for col, dumped in zip(config.columns, columns_dump):
        if isinstance(col, CustomColumnConfig):
            fn = col.generator_function
            metadata = getattr(fn, "custom_column_metadata", {}) or {}
            dumped = {
                **dumped,
                "generator_function": {
                    "name": getattr(fn, "__name__", None),
                    "qualname": getattr(fn, "__qualname__", None),
                    "module": getattr(fn, "__module__", None),
                    "metadata": metadata,
                },
            }
        enriched.append(dumped)
    return enriched


def _normalize_config_dict(config_dict: dict[str, Any], config: DataDesignerConfig) -> dict[str, Any]:
    normalized = _drop_keys(config_dict, _EXCLUDED_TOP_LEVEL_KEYS)
    normalized = _drop_empty_optional(normalized, _TOP_LEVEL_OPTIONAL_COLLECTIONS)

    columns = normalized.get("columns")
    if columns:
        normalized["columns"] = _enrich_custom_columns(config, columns)

    model_configs = normalized.get("model_configs")
    if model_configs:
        normalized["model_configs"] = sorted(
            (_normalize_model_config(mc) for mc in model_configs),
            key=lambda mc: mc.get("alias", ""),
        )

    tool_configs = normalized.get("tool_configs")
    if tool_configs:
        normalized["tool_configs"] = sorted(
            (_normalize_tool_config(tc) for tc in tool_configs),
            key=lambda tc: tc.get("tool_alias", ""),
        )

    seed_config = normalized.get("seed_config")
    if seed_config:
        normalized["seed_config"] = _normalize_seed_config(seed_config)

    return normalized
