# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from data_designer.engine.column_generators.generators.base import ColumnGenerator

logger = logging.getLogger(__name__)

SchedulingGroupKind = Literal["local", "model", "custom_model"]


@dataclass(frozen=True)
class SchedulingHint:
    """Resolved task-scheduling metadata independent of graph flow identity."""

    group_kind: SchedulingGroupKind
    identity_prefix: tuple[str, ...] = ()
    identity_suffix: tuple[str, ...] = ()
    weight: int = 1


class SchedulingHintResolver:
    """Resolve generator/config/model metadata once for a scheduler run."""

    def __init__(self, generators: dict[str, ColumnGenerator]) -> None:
        self._hints_by_generator_id: dict[int, SchedulingHint] = {}
        for column, generator in generators.items():
            generator_id = id(generator)
            if generator_id not in self._hints_by_generator_id:
                self._hints_by_generator_id[generator_id] = self._resolve_hint(column, generator)

    def hint_for(self, generator: ColumnGenerator) -> SchedulingHint:
        return self._hints_by_generator_id[id(generator)]

    def _resolve_hint(self, column: str, generator: ColumnGenerator) -> SchedulingHint:
        if not generator.is_llm_bound:
            return SchedulingHint(group_kind="local")

        aliases = _model_aliases_for_generator(generator)
        if not aliases:
            return SchedulingHint(group_kind="model", identity_prefix=("unknown",), weight=1)

        model_parts: list[str] = []
        total_parallel = 0
        primary_alias = getattr(generator.config, "model_alias", None)
        for alias in aliases:
            try:
                model_config = _get_model_config_for_alias(generator, alias)
                provider_name = _get_model_provider_name_for_alias(generator, alias)
            except Exception:
                logger.debug(
                    "Falling back to custom-model scheduling group for column %r after failing to resolve "
                    "model alias %r from aliases %r.",
                    column,
                    alias,
                    aliases,
                    exc_info=True,
                )
                return SchedulingHint(
                    group_kind="custom_model",
                    identity_suffix=tuple(sorted(aliases)),
                    weight=max(1, total_parallel),
                )

            max_parallel = getattr(model_config.inference_parameters, "max_parallel_requests", 1)
            if not isinstance(max_parallel, int):
                max_parallel = 1
            model_parts.extend(
                (
                    provider_name,
                    str(model_config.model),
                    str(model_config.generation_type),
                    alias,
                )
            )
            total_parallel += max_parallel

        weight = max(1, total_parallel)
        if len(aliases) == 1 and primary_alias == aliases[0]:
            return SchedulingHint(
                group_kind="model",
                identity_prefix=tuple(model_parts[:3]),
                weight=weight,
            )

        return SchedulingHint(
            group_kind="custom_model",
            identity_suffix=tuple(sorted(aliases)),
            weight=weight,
        )


def _get_model_config_for_alias(generator: ColumnGenerator, alias: str) -> Any:
    get_model_config = getattr(generator, "get_model_config", None)
    if callable(get_model_config):
        return get_model_config(model_alias=alias)
    return generator.resource_provider.model_registry.get_model_config(model_alias=alias)


def _get_model_provider_name_for_alias(generator: ColumnGenerator, alias: str) -> str:
    get_provider_name = getattr(generator, "get_model_provider_name", None)
    if callable(get_provider_name):
        return str(get_provider_name(model_alias=alias))
    provider = generator.resource_provider.model_registry.get_model_provider(model_alias=alias)
    return str(provider.name)


def _model_aliases_for_generator(generator: ColumnGenerator) -> list[str]:
    get_aliases = getattr(generator.config, "get_model_aliases", None)
    if callable(get_aliases):
        aliases = get_aliases()
    else:
        aliases = []
        if (alias := getattr(generator.config, "model_alias", None)) is not None:
            aliases.append(alias)
        aliases.extend(getattr(generator.config, "model_aliases", []) or [])
    return list(dict.fromkeys(alias for alias in aliases if alias))
