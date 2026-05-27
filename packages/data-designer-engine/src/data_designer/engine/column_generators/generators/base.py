# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar, overload

from data_designer.config.column_configs import GenerationStrategy
from data_designer.config.scheduling import SchedulingMetadata, SchedulingMetadataError
from data_designer.engine.configurable_task import ConfigurableTask, DataT, TaskConfigT
from data_designer.logging import LOG_DOUBLE_INDENT, LOG_INDENT

_T = TypeVar("_T")

# Preserved deliberately. Two other 300s deadlines were retired in the
# async-default flip (PR #592): the request-admission queue wait and the
# ``_AsyncBridgedModelFacade`` bridge in ``custom.py`` — both have
# ``ModelFacade`` context and could derive a per-call deadline from
# ``inference_parameters.timeout``. This generic ``ColumnGenerator.generate()``
# fallback has no facade reachable, so a defensive backstop stays here for
# now. Wiring a per-call timeout through to ``_run_coroutine_sync`` is
# tracked as a structural follow-up.
SYNC_BRIDGE_TIMEOUT = 300


@dataclass
class _EndpointBucket:
    aliases: list[str] = field(default_factory=list)
    caps: list[int] = field(default_factory=list)


def _scheduling_generation_kind(generation_type: object) -> str:
    value = getattr(generation_type, "value", generation_type)
    if value == "chat-completion":
        return "chat"
    return str(value)


if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.models import BaseInferenceParams, ModelConfig
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def _run_coroutine_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine from sync context.

    - No running event loop → ``asyncio.run(coro)``
    - Running event loop (e.g. notebook/service) → run in a background thread
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(asyncio.run, coro)
    timed_out = False
    try:
        result = future.result(timeout=SYNC_BRIDGE_TIMEOUT)
    except concurrent.futures.TimeoutError as exc:
        timed_out = True
        logger.warning(f"⚠️ Sync bridge timed out after {SYNC_BRIDGE_TIMEOUT}s; background thread still running")
        raise TimeoutError(f"_run_coroutine_sync timed out after {SYNC_BRIDGE_TIMEOUT}s") from exc
    finally:
        pool.shutdown(wait=not timed_out, cancel_futures=timed_out)
    return result


class ColumnGenerator(ConfigurableTask[TaskConfigT], ABC):
    @property
    def can_generate_from_scratch(self) -> bool:
        return False

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        """Return static scheduler metadata for this generator.

        Generators that do not declare model-backed behavior use the documented
        local default. Model-aware base classes override this with provider/model
        resource identity derived from registered model aliases.
        """
        return SchedulingMetadata.local()

    @property
    def is_order_dependent(self) -> bool:
        """Whether this generator's output depends on prior row-group calls.

        Example: SeedDatasetColumnGenerator tracks its position in the seed
        dataset, so row group N must complete before N+1 starts.
        """
        return False

    def _is_overridden(self, method_name: str) -> bool:
        """Check if a subclass has overridden a base ColumnGenerator method."""
        return getattr(type(self), method_name) is not getattr(ColumnGenerator, method_name)

    @staticmethod
    @abstractmethod
    def get_generation_strategy() -> GenerationStrategy: ...

    @overload
    def generate(self, data: dict) -> dict: ...

    @overload
    def generate(self, data: pd.DataFrame) -> pd.DataFrame: ...

    def generate(self, data: DataT) -> DataT:
        """Sync generate — overridden by most concrete generators.

        Default bridges to ``agenerate()`` for async-first subclasses that only
        implement ``agenerate()``. Raises ``NotImplementedError`` if neither
        ``generate()`` nor ``agenerate()`` is overridden.
        """
        if not self._is_overridden("agenerate"):
            raise NotImplementedError(f"{type(self).__name__} must implement either generate() or agenerate()")
        return _run_coroutine_sync(self.agenerate(data))

    @overload
    async def agenerate(self, data: dict) -> dict: ...

    @overload
    async def agenerate(self, data: pd.DataFrame) -> pd.DataFrame: ...

    async def agenerate(self, data: DataT) -> DataT:
        """Async generate — delegates to sync ``generate()`` via thread pool.

        Subclasses with native async support (e.g. ColumnGeneratorWithModelChatCompletion)
        should override this with a direct async implementation.
        """
        if not self._is_overridden("generate"):
            raise NotImplementedError(f"{type(self).__name__} must implement either generate() or agenerate()")
        return await asyncio.to_thread(self.generate, data.copy())

    def log_pre_generation(self) -> None:
        """A shared method to log info before the generator's `generate` method is called.

        The idea is for dataset builders to call this method for all generators before calling their
        `generate` method. This is to avoid logging the same information multiple times when running
        generators in parallel.
        """


class FromScratchColumnGenerator(ColumnGenerator[TaskConfigT], ABC):
    @property
    def can_generate_from_scratch(self) -> bool:
        return True

    @abstractmethod
    def generate_from_scratch(self, num_records: int) -> pd.DataFrame: ...

    async def agenerate_from_scratch(self, num_records: int) -> pd.DataFrame:
        """Async wrapper — wraps sync ``generate_from_scratch()`` in a thread."""
        return await asyncio.to_thread(self.generate_from_scratch, num_records)


class ColumnGeneratorWithModelRegistry(ColumnGenerator[TaskConfigT], ABC):
    @property
    def model_registry(self) -> ModelRegistry:
        return self.resource_provider.model_registry

    def get_model(self, model_alias: str) -> ModelFacade:
        return self.model_registry.get_model(model_alias=model_alias)

    def get_model_config(self, model_alias: str) -> ModelConfig:
        return self.model_registry.get_model_config(model_alias=model_alias)

    def get_model_provider_name(self, model_alias: str) -> str:
        provider = self.model_registry.get_model_provider(model_alias=model_alias)
        return provider.name

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        aliases = self._get_scheduling_model_aliases()
        if not aliases:
            raise SchedulingMetadataError(
                code="missing_model_alias",
                message=f"{type(self).__name__} has no model aliases for scheduling metadata.",
                fallback=SchedulingMetadata.local(),
                diagnostics={"generator_type": type(self).__name__},
            )

        endpoints: dict[tuple[str, str, str], _EndpointBucket] = {}
        for alias in aliases:
            try:
                model_config = self.get_model_config(model_alias=alias)
                provider_name = self.get_model_provider_name(model_alias=alias)
            except Exception as exc:
                raise SchedulingMetadataError(
                    code="alias_resolution_failed",
                    message=f"Could not resolve model alias {alias!r} for scheduling metadata.",
                    diagnostics={"alias": alias, "generator_type": type(self).__name__},
                ) from exc

            endpoint = (
                provider_name,
                str(model_config.model),
                _scheduling_generation_kind(model_config.generation_type),
            )
            max_parallel = getattr(model_config.inference_parameters, "max_parallel_requests", 1)
            cap = max_parallel if isinstance(max_parallel, int) and max_parallel > 0 else 1
            bucket = endpoints.setdefault(endpoint, _EndpointBucket())
            bucket.aliases.append(alias)
            bucket.caps.append(cap)

        if len(endpoints) != 1:
            raw_caps = tuple(cap for bucket in endpoints.values() for cap in bucket.caps)
            return SchedulingMetadata.custom_model(
                _scheduling_plugin_namespace(type(self)),
                _scheduling_alias_set_resource_name(aliases),
                "v1",
                weight=max(1, sum(raw_caps)),
                diagnostics={
                    "aliases": tuple(sorted(aliases)),
                    "endpoints": tuple(sorted(str(endpoint) for endpoint in endpoints)),
                    "fallback_reason": "multi_endpoint_alias_set",
                    "raw_caps": raw_caps,
                },
            )

        endpoint, bucket = next(iter(endpoints.items()))
        provider_name, model_id, generation_kind = endpoint
        effective_cap = max(1, min(bucket.caps))
        return SchedulingMetadata.model(
            provider_name,
            model_id,
            generation_kind,
            weight=effective_cap,
            diagnostics={
                "aliases": tuple(bucket.aliases),
                "raw_caps": tuple(bucket.caps),
                "merge_rule": "min_same_endpoint",
            },
        )

    def _get_scheduling_model_aliases(self) -> list[str]:
        get_aliases = getattr(self.config, "get_model_aliases", None)
        if callable(get_aliases):
            aliases = get_aliases()
        else:
            aliases = []
            if (alias := getattr(self.config, "model_alias", None)) is not None:
                aliases.append(alias)
            aliases.extend(getattr(self.config, "model_aliases", []) or [])
        return list(dict.fromkeys(str(alias) for alias in aliases if alias))


def _scheduling_plugin_namespace(generator_type: type[object]) -> str:
    return f"{generator_type.__module__}.{generator_type.__qualname__}"


def _scheduling_alias_set_resource_name(aliases: list[str]) -> str:
    alias_key = "\0".join(sorted(aliases)).encode()
    digest = hashlib.sha1(alias_key).hexdigest()[:16]
    return f"alias-set-{digest}"


class ColumnGeneratorWithModel(ColumnGeneratorWithModelRegistry[TaskConfigT], ABC):
    @functools.cached_property
    def model(self) -> ModelFacade:
        return self.get_model(model_alias=self.config.model_alias)

    @functools.cached_property
    def model_config(self) -> ModelConfig:
        return self.get_model_config(model_alias=self.config.model_alias)

    @functools.cached_property
    def inference_parameters(self) -> BaseInferenceParams:
        return self.model_config.inference_parameters

    def _build_multi_modal_context(self, record: dict) -> list[dict[str, Any]] | None:
        """Build multi-modal context from the config's multi_modal_context list.

        Passes base_path to get_contexts() so context types that support
        artifact-relative resolution can use the dataset artifact directory.

        Args:
            record: The deserialized record containing column values.

        Returns:
            A list of multi-modal context dicts, or None if no context is configured.
        """
        if not hasattr(self.config, "multi_modal_context") or not self.config.multi_modal_context:
            return None
        base_path = str(self.base_dataset_path)
        multi_modal_context: list[dict[str, Any]] = []
        for context in self.config.multi_modal_context:
            multi_modal_context.extend(context.get_contexts(record, base_path=base_path))
        return multi_modal_context

    def log_pre_generation(self) -> None:
        logger.info(
            f"{self.config.get_column_emoji()} {self.config.column_type} model config for column '{self.config.name}'"
        )
        logger.info(f"{LOG_INDENT}model: {self.model_config.model!r}")
        logger.info(f"{LOG_INDENT}model alias: {self.config.model_alias!r}")
        logger.info(
            f"{LOG_INDENT}model provider: {self.get_model_provider_name(model_alias=self.config.model_alias)!r}"
        )
        logger.info(f"{LOG_INDENT}inference parameters:")
        for param in self.inference_parameters.get_formatted_params():
            logger.info(f"{LOG_DOUBLE_INDENT}{param}")

        tool_alias = getattr(self.config, "tool_alias", None)
        if tool_alias is not None:
            tool_config = self.resource_provider.mcp_registry.get_tool_config(tool_alias=tool_alias)
            logger.info(f"{LOG_INDENT}tool alias: {tool_alias!r}")
            logger.info(f"{LOG_INDENT}mcp providers: {tool_config.providers!r}")


class ColumnGeneratorCellByCell(ColumnGenerator[TaskConfigT], ABC):
    """Base class for column generators invoked once per row.

    Override ``generate`` to return the complete row mapping after adding the
    generated value. The engine calls the generator once per row and may run
    calls concurrently. Use this base when generation is independent per row
    (e.g. an LLM call per row, a per-row transform).
    """

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    @abstractmethod
    def generate(self, data: dict) -> dict:
        """Generate one row's output from a single row's upstream values.

        Args:
            data: Current row mapping containing the upstream values available to this column.

        Returns:
            Complete row mapping with existing keys preserved and the new column value added.
            Include declared side-effect columns when the config creates them.
        """


class ColumnGeneratorFullColumn(ColumnGenerator[TaskConfigT], ABC):
    """Base class for column generators that transform a full batch at once.

    Override ``generate`` to return the complete batch DataFrame after adding
    generated values. Use this base when generation is vectorizable or when an
    external API accepts batched input more efficiently than per-row calls.
    """

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate an entire batch of row outputs.

        Args:
            data: DataFrame containing the upstream columns this generator depends on.

        Returns:
            DataFrame containing the input columns plus the new column and any side-effect
            columns. When ``config.allow_resize`` is ``False``, the row count must match
            the input; when it is ``True``, the row count may change.
        """
