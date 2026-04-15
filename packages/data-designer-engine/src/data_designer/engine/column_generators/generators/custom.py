# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom column generator using user-provided callable functions."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig, GenerationStrategy
from data_designer.engine.column_generators.generators.base import _SYNC_BRIDGE_TIMEOUT, ColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.logging import LOG_INDENT

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class _AsyncBridgedModelFacade:
    """Proxy that bridges ``model.generate()`` to ``model.agenerate()`` in async engine mode.

    When a sync custom column runs inside ``asyncio.to_thread`` under the async engine,
    the sync HTTP client is unavailable. This proxy intercepts the resulting
    ``SyncClientUnavailableError`` and schedules ``agenerate()`` on the engine's persistent
    event loop via ``run_coroutine_threadsafe``.

    All other attributes are forwarded to the underlying facade unchanged.
    """

    __slots__ = ("_facade",)

    def __init__(self, facade: Any) -> None:
        object.__setattr__(self, "_facade", facade)

    def generate(self, *args: Any, **kwargs: Any) -> tuple[Any, list]:
        from data_designer.engine.models.clients.errors import SyncClientUnavailableError

        facade = object.__getattribute__(self, "_facade")
        try:
            return facade.generate(*args, **kwargs)
        except SyncClientUnavailableError:
            pass  # Fall through to async bridge

        # We're in a worker thread (asyncio.to_thread) with no running loop.
        # Guard against accidental use from the event loop itself (would deadlock).
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop - safe to bridge
        else:
            raise RuntimeError(
                "model.generate() is not available in async engine mode from the event loop. "
                "Use 'await model.agenerate()' in async custom columns."
            )

        from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(facade.agenerate(*args, **kwargs), loop)
        try:
            return future.result(timeout=_SYNC_BRIDGE_TIMEOUT)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            logger.warning("Async model bridge timed out after %ss; coroutine cancelled", _SYNC_BRIDGE_TIMEOUT)
            raise TimeoutError(f"model.generate() bridge timed out after {_SYNC_BRIDGE_TIMEOUT}s") from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_facade"), name)

    def __repr__(self) -> str:
        return f"_AsyncBridgedModelFacade({object.__getattribute__(self, '_facade')!r})"


class CustomColumnGenerator(ColumnGenerator[CustomColumnConfig]):
    """Column generator that uses a user-provided callable function.

    Supports two strategies based on config.strategy:
        - cell_by_cell: Processes rows one at a time (dict -> dict), parallelized by framework.
        - full_column: Processes entire batch (DataFrame -> DataFrame) for vectorized ops.

    Supported function signatures (validated by parameter name):
        - fn(row) -> dict                              # cell_by_cell, simple transform
        - fn(row, generator_params) -> dict            # cell_by_cell, with typed params
        - fn(row, generator_params, models) -> dict    # cell_by_cell, with LLM access
        - fn(df) -> DataFrame                          # full_column, simple transform
        - fn(df, generator_params) -> DataFrame        # full_column, with typed params
        - fn(df, generator_params, models) -> DataFrame  # full_column, with LLM access

    The models dict provides direct access to ModelFacade instances keyed by alias.
    """

    @property
    def is_llm_bound(self) -> bool:
        """Custom generators with model_aliases make LLM calls and need the handoff."""
        return bool(self.config.model_aliases)

    def get_generation_strategy(self) -> GenerationStrategy:
        """Return strategy based on config."""
        return self.config.generation_strategy

    def generate(self, data: dict | pd.DataFrame) -> dict | pd.DataFrame | list[dict]:
        """Generate column value(s) for a row (dict) or batch (DataFrame).

        For cell_by_cell with allow_resize=True, may return dict or list[dict] (0, 1, or N rows).
        """
        is_full_column = self.config.generation_strategy == GenerationStrategy.FULL_COLUMN
        is_dataframe = not isinstance(data, dict)

        # Validate data type matches strategy
        if is_full_column and not is_dataframe:
            raise CustomColumnGenerationError(
                f"Custom generator {self.config.name!r} is configured for 'full_column' strategy "
                "but received a dict. Expected a DataFrame."
            )
        if not is_full_column and is_dataframe:
            raise CustomColumnGenerationError(
                f"Custom generator {self.config.name!r} is configured for 'cell_by_cell' strategy "
                "but received a DataFrame. Expected a dict."
            )

        return self._generate(data, is_dataframe)

    async def agenerate(self, data: dict | pd.DataFrame) -> dict | pd.DataFrame | list[dict]:
        """Async generate — branches on strategy and detects coroutine functions."""
        is_full_column = self.config.generation_strategy == GenerationStrategy.FULL_COLUMN
        if is_full_column:
            return await asyncio.to_thread(self.generate, data.copy())
        # The @custom_column_generator decorator wraps the user function in a sync
        # wrapper, so we must unwrap to detect async functions.
        fn_unwrapped = inspect.unwrap(self.config.generator_function)
        if asyncio.iscoroutinefunction(fn_unwrapped):
            missing = set(self.config.required_columns) - set(data.keys())
            if missing:
                raise CustomColumnGenerationError(
                    f"Missing required columns for custom generator '{self.config.name}': {sorted(missing)}"
                )
            keys_before = set(data.keys())

            try:
                result = await self._ainvoke_generator_function(data)
            except CustomColumnGenerationError:
                raise
            except Exception as e:
                logger.warning(
                    f"⚠️ Custom generator function {self.config.generator_function.__name__!r} "
                    f"failed for column '{self.config.name}'. This record will be skipped.\n{e}"
                )
                raise CustomColumnGenerationError(
                    f"Custom generator function failed for column '{self.config.name}': {e}"
                ) from e

            return self._postprocess_result(result, is_dataframe=False, keys_before=keys_before)
        return await asyncio.to_thread(self.generate, data)

    async def _ainvoke_generator_function(self, data: dict) -> dict | pd.DataFrame:
        """Invoke an async user generator function with appropriate arguments.

        The @custom_column_generator decorator's sync wrapper returns a coroutine
        when the original function is async, so we await the wrapper's return value.
        """
        params = self._get_validated_params()
        fn = self.config.generator_function
        if len(params) == 1:
            return await fn(data)
        elif len(params) == 2:
            return await fn(data, self.config.generator_params)
        else:
            models = self._build_models_dict()
            return await fn(data, self.config.generator_params, models)

    def _generate(self, data: dict | pd.DataFrame, is_dataframe: bool) -> dict | pd.DataFrame | list[dict]:
        """Unified generation logic for both strategies."""
        get_keys = (lambda d: set(d.columns)) if is_dataframe else (lambda d: set(d.keys()))

        # Check required columns
        missing = set(self.config.required_columns) - get_keys(data)
        if missing:
            raise CustomColumnGenerationError(
                f"Missing required columns for custom generator '{self.config.name}': {sorted(missing)}"
            )

        keys_before = get_keys(data)

        # Invoke generator
        try:
            result = self._invoke_generator_function(data)
        except CustomColumnGenerationError:
            raise
        except Exception as e:
            if not is_dataframe:
                logger.warning(
                    f"⚠️ Custom generator function {self.config.generator_function.__name__!r} "
                    f"failed for column '{self.config.name}'. This record will be skipped.\n{e}"
                )
            raise CustomColumnGenerationError(
                f"Custom generator function failed for column '{self.config.name}': {e}"
            ) from e

        return self._postprocess_result(result, is_dataframe, keys_before)

    def _postprocess_result(
        self,
        result: dict | pd.DataFrame | list[dict],
        is_dataframe: bool,
        keys_before: set[str],
    ) -> dict | pd.DataFrame | list[dict]:
        """Validate type and output columns of a generation result."""
        # Cell-by-cell with allow_resize: accept dict or list[dict]
        if not is_dataframe and self.config.allow_resize:
            if isinstance(result, dict):
                return self._validate_output(result, keys_before, is_dataframe)
            if isinstance(result, list):
                if not all(isinstance(r, dict) for r in result):
                    raise CustomColumnGenerationError(
                        f"Custom generator for column '{self.config.name}' with allow_resize must return "
                        "dict or list[dict]; list elements must be dicts."
                    )
                return [self._validate_cell_output(r, keys_before) for r in result]
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' with allow_resize must return "
                f"dict or list[dict], got {type(result).__name__}"
            )

        # Validate return type for non-resize paths
        expected_type = lazy.pd.DataFrame if is_dataframe else dict
        type_name = "DataFrame" if is_dataframe else "dict"
        if not isinstance(result, expected_type):
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' must return a {type_name}, "
                f"got {type(result).__name__}"
            )

        return self._validate_output(result, keys_before, is_dataframe)

    def _validate_cell_output(self, row: dict, keys_before: set[str]) -> dict:
        """Validate a single row output (dict) for cell_by_cell; strip undeclared columns."""
        expected_new = {self.config.name} | set(self.config.side_effect_columns)
        result_keys = set(row.keys())

        if self.config.name not in result_keys:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create the expected column. "
                f"The generator_function must add a column named '{self.config.name}' to the row."
            )
        missing = set(self.config.side_effect_columns) - result_keys
        if missing:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create declared side_effect_columns: "
                f"{sorted(missing)}. Declared side_effect_columns must be added to the row."
            )
        removed = keys_before - result_keys
        if removed:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' removed pre-existing columns: "
                f"{sorted(removed)}. The generator_function must not remove any existing columns."
            )
        undeclared = (result_keys - keys_before) - expected_new
        if undeclared:
            logger.warning(
                f"⚠️ Custom generator for column '{self.config.name}' created undeclared columns: "
                f"{sorted(undeclared)}. These columns will be removed. "
                f"To keep additional columns, declare them in @custom_column_generator(side_effect_columns=[...])."
            )
            row = {k: v for k, v in row.items() if k not in undeclared}
        return row

    def _validate_output(
        self, result: dict | pd.DataFrame, keys_before: set[str], is_dataframe: bool
    ) -> dict | pd.DataFrame:
        """Validate output columns and remove undeclared ones."""
        # Unified accessors
        get_keys = (lambda d: set(d.columns)) if is_dataframe else (lambda d: set(d.keys()))
        container_name = "DataFrame" if is_dataframe else "row"

        expected_new = {self.config.name} | set(self.config.side_effect_columns)
        result_keys = get_keys(result)

        # Check primary column exists
        if self.config.name not in result_keys:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create the expected column. "
                f"The generator_function must add a key named '{self.config.name}' to the {container_name}."
            )

        # Check side effect columns exist
        missing = set(self.config.side_effect_columns) - result_keys
        if missing:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' did not create declared side_effect_columns: "
                f"{sorted(missing)}. Declared side_effect_columns must be added to the {container_name}."
            )

        # Check no pre-existing columns removed
        removed = keys_before - result_keys
        if removed:
            raise CustomColumnGenerationError(
                f"Custom generator for column '{self.config.name}' removed pre-existing columns: "
                f"{sorted(removed)}. The generator_function must not remove any existing columns."
            )

        # Remove undeclared columns with warning
        undeclared = (result_keys - keys_before) - expected_new
        if undeclared:
            logger.warning(
                f"⚠️ Custom generator for column '{self.config.name}' created undeclared columns: "
                f"{sorted(undeclared)}. These columns will be removed. "
                f"To keep additional columns, declare them in @custom_column_generator(side_effect_columns=[...])."
            )
            if is_dataframe:
                result = result.drop(columns=list(undeclared))
            else:
                result = {k: v for k, v in result.items() if k not in undeclared}

        return result

    def _invoke_generator_function(self, data: dict | pd.DataFrame) -> dict | pd.DataFrame:
        """Invoke the user's generate function with appropriate arguments based on signature."""
        params = self._get_validated_params()

        if len(params) == 1:
            return self.config.generator_function(data)
        elif len(params) == 2:
            return self.config.generator_function(data, self.config.generator_params)
        else:
            models = {k: _AsyncBridgedModelFacade(v) for k, v in self._build_models_dict().items()}
            return self.config.generator_function(data, self.config.generator_params, models)

    def _build_models_dict(self) -> dict[str, Any]:
        """Build a dict of ModelFacade instances from model_aliases."""
        return {
            alias: self.resource_provider.model_registry.get_model(model_alias=alias)
            for alias in self.config.model_aliases
        }

    def _get_validated_params(self) -> list[inspect.Parameter]:
        """Get positional params and validate first param matches generation strategy."""
        params = [
            p
            for p in inspect.signature(self.config.generator_function).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        # Decorator validated param names; here we only check strategy match
        is_full = self.config.generation_strategy == GenerationStrategy.FULL_COLUMN
        expected = "df" if is_full else "row"
        if params[0].name != expected:
            raise CustomColumnGenerationError(
                f"Generator '{self.config.name}': strategy is {'full_column' if is_full else 'cell_by_cell'}, "
                f"first parameter must be '{expected}', got '{params[0].name}'."
            )
        return params

    def log_pre_generation(self) -> None:
        logger.info(f"{self.config.get_column_emoji()} Custom column config for column '{self.config.name}'")
        logger.info(f"{LOG_INDENT}generator_function: {self.config.generator_function.__name__!r}")
        logger.info(f"{LOG_INDENT}generation_strategy: {self.config.generation_strategy!r}")
        logger.info(f"{LOG_INDENT}required_columns: {self.config.required_columns}")
        if self.config.side_effect_columns:
            logger.info(f"{LOG_INDENT}side_effect_columns: {self.config.side_effect_columns}")
        if self.config.model_aliases:
            logger.info(f"{LOG_INDENT}model_aliases: {self.config.model_aliases}")
        if self.config.generator_params:
            logger.info(f"{LOG_INDENT}generator_params: {self.config.generator_params}")
        if self.config.allow_resize:
            logger.info(f"{LOG_INDENT}allow_resize: {self.config.allow_resize}")
