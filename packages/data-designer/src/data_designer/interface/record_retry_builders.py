# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType, is_plugin_column_type
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.processor_types import ProcessorConfigT
from data_designer.config.processors import DropColumnsProcessorConfig, ProcessorType
from data_designer.config.sampler_constraints import ColumnInequalityConstraint
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.sampling_gen.jinja_utils import extract_column_names_from_expression
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.record_retry import RetryUntil, SamplerRetryMode

_IMPLICIT_DROP_PROCESSOR_BASENAME = "_data_designer_record_retry_implicit_drop"
_SEED_PASSTHROUGH_PROCESSOR_BASENAME = "_data_designer_record_retry_seed_passthrough"
_SAMPLER_REFERENCE_PARAM_FIELDS = ("category", "reference_column_name")


class RecordRetryBuilderFactory:
    """Create isolated builders for the three record-retry execution phases.

    The factory owns a deep snapshot of the user's builder. Hidden base and
    attempt runs never mutate that builder, never run its processors or
    profilers, and force selected columns to remain materialized for durable
    classification. The final builder contains no generators: it reads the
    coalesced accepted rows as a seed and applies the original terminal
    processors and profilers once.
    """

    original_config: DataDesignerConfig
    retry_until: RetryUntil
    requires_base_materialization: bool

    def __init__(self, config_builder: DataDesignerConfigBuilder, retry_until: RetryUntil) -> None:
        original_config = config_builder.build().model_copy(deep=True)
        validate_record_retry_predicate(original_config, retry_until.predicate_column)

        generated_columns = tuple(
            column
            for column in original_config.columns
            if column.column_type not in {DataDesignerColumnType.SEED_DATASET, DataDesignerColumnType.SAMPLER}
        )
        if retry_until.sampler_retry_mode == SamplerRetryMode.PRESERVE:
            _validate_preserved_sampler_dependencies(original_config, generated_columns)

        has_preserved_sampler = retry_until.sampler_retry_mode == SamplerRetryMode.PRESERVE and any(
            column.column_type == DataDesignerColumnType.SAMPLER for column in original_config.columns
        )
        self.original_config = original_config
        self.retry_until = retry_until
        self.requires_base_materialization = original_config.seed_config is not None or has_preserved_sampler

    def build_original_builder(self) -> DataDesignerConfigBuilder:
        """Reconstruct the user's original declarative builder snapshot."""
        return DataDesignerConfigBuilder.from_config(
            BuilderConfig(data_designer=self.original_config.model_copy(deep=True))
        )

    def build_base_builder(self) -> DataDesignerConfigBuilder:
        """Build the immutable base-cohort materialization config."""
        builder = _empty_builder(self.original_config)
        if self.original_config.seed_config is not None:
            seed_config = self.original_config.seed_config.model_copy(deep=True)
            builder.with_seed_dataset(
                seed_config.source,
                sampling_strategy=seed_config.sampling_strategy,
                selection_strategy=seed_config.selection_strategy,
            )

        if self.retry_until.sampler_retry_mode == SamplerRetryMode.PRESERVE:
            for column in self.original_config.columns:
                if column.column_type == DataDesignerColumnType.SAMPLER:
                    builder.add_column(_hidden_column_copy(column))
            for constraint in self.original_config.constraints or []:
                builder.add_constraint(constraint.model_copy(deep=True))

        if not builder.get_column_configs():
            _add_seed_passthrough_processor(builder, self.original_config)

        return builder

    def build_attempt_builder(self, input_path: Path | str) -> DataDesignerConfigBuilder:
        """Build one generation attempt over an exact pending-slot parquet projection."""
        builder = _empty_builder(self.original_config)
        builder.with_seed_dataset(_local_file_seed_source(input_path), sampling_strategy=SamplingStrategy.ORDERED)

        for column in self.original_config.columns:
            if column.column_type == DataDesignerColumnType.SEED_DATASET:
                continue
            if column.column_type == DataDesignerColumnType.SAMPLER:
                if self.retry_until.sampler_retry_mode == SamplerRetryMode.RESAMPLE:
                    builder.add_column(_hidden_column_copy(column))
                continue
            builder.add_column(_hidden_column_copy(column))

        if self.retry_until.sampler_retry_mode == SamplerRetryMode.RESAMPLE:
            for constraint in self.original_config.constraints or []:
                builder.add_constraint(constraint.model_copy(deep=True))

        return builder

    def build_final_builder(self, input_path: Path | str) -> DataDesignerConfigBuilder:
        """Build the accepted-only terminal processor and profiler pass."""
        builder = _empty_builder(self.original_config)
        builder.with_seed_dataset(_local_file_seed_source(input_path), sampling_strategy=SamplingStrategy.ORDERED)

        original_processors = self.original_config.processors or []
        for processor in original_processors:
            builder.add_processor(processor.model_copy(deep=True))

        explicitly_dropped = _explicitly_dropped_configured_columns(
            columns=self.original_config.columns,
            processors=original_processors,
        )
        implicit_drop_names = [
            column.name
            for column in self.original_config.columns
            if column.drop and column.name not in explicitly_dropped
        ]
        if implicit_drop_names:
            builder.add_processor(
                DropColumnsProcessorConfig(
                    name=unique_name(
                        _IMPLICIT_DROP_PROCESSOR_BASENAME,
                        {processor.name for processor in builder.get_processor_configs()},
                    ),
                    column_names=implicit_drop_names,
                )
            )

        if not builder.get_processor_configs():
            _add_seed_passthrough_processor(builder, self.original_config)

        for profiler in self.original_config.profilers or []:
            builder.add_profiler(profiler.model_copy(deep=True))

        return builder


def unique_name(base_name: str, used_names: set[str]) -> str:
    """Return a deterministic unused name based on the supplied base."""
    if base_name not in used_names:
        return base_name
    suffix = 1
    while f"{base_name}_{suffix}" in used_names:
        suffix += 1
    return f"{base_name}_{suffix}"


def validate_record_retry_predicate(config: DataDesignerConfig, predicate_column: str) -> None:
    """Validate that the predicate is declared by a rerunnable generator.

    Expression predicates must explicitly declare ``dtype="bool"``. Custom,
    plugin, and LLM-text outputs remain subject to the runner's strict scalar
    bool/null check. The built-in LLM-text generator always returns strings, so
    a direct LLM-text predicate cannot pass that runtime contract today; add a
    Boolean expression column over the text instead. Strings such as ``"true"``
    and ``"false"`` are never coerced. Structured output, including a nested
    Boolean, must likewise feed a separate Boolean expression predicate.
    """
    owners = [
        column
        for column in config.columns
        if column.name == predicate_column or predicate_column in column.side_effect_columns
    ]
    if not owners:
        raise DataDesignerWorkflowError(
            f"retry_until.predicate_column {predicate_column!r} is not declared by any configured column."
        )
    if len(owners) > 1:
        raise DataDesignerWorkflowError(
            f"retry_until.predicate_column {predicate_column!r} has more than one declared owner."
        )

    predicate_owner = owners[0]
    column_type = predicate_owner.column_type
    if column_type == DataDesignerColumnType.EXPRESSION:
        if predicate_owner.name == predicate_column and getattr(predicate_owner, "dtype", None) == "bool":
            return
        raise DataDesignerWorkflowError(f"retry_until predicate expression {predicate_column!r} must use dtype='bool'.")

    if column_type in {DataDesignerColumnType.CUSTOM, DataDesignerColumnType.LLM_TEXT}:
        return
    if is_plugin_column_type(column_type):
        return

    raise DataDesignerWorkflowError(
        f"retry_until.predicate_column {predicate_column!r} uses unsupported column type {str(column_type)!r}; "
        "expected a Boolean expression, custom column, plugin column, or runtime-Boolean LLM-text column. "
        "For nested structured output, extract the Boolean into an expression column."
    )


def _empty_builder(config: DataDesignerConfig) -> DataDesignerConfigBuilder:
    model_configs = [model.model_copy(deep=True) for model in config.model_configs or []]
    if not model_configs:
        raise DataDesignerWorkflowError("Record retry requires the original builder to contain model configs.")
    return DataDesignerConfigBuilder(
        model_configs=model_configs,
        tool_configs=[tool.model_copy(deep=True) for tool in config.tool_configs or []],
    )


def _hidden_column_copy(column: ColumnConfigT) -> ColumnConfigT:
    copied = column.model_copy(deep=True)
    copied.drop = False
    return copied


def _local_file_seed_source(path: Path | str) -> LocalFileSeedSource:
    path = Path(path)
    source_path = path / "*.parquet" if path.is_dir() else path
    return LocalFileSeedSource(path=str(source_path))


def _add_seed_passthrough_processor(
    builder: DataDesignerConfigBuilder,
    original_config: DataDesignerConfig,
) -> None:
    existing_names = {processor.name for processor in original_config.processors or []}
    builder.add_processor(
        DropColumnsProcessorConfig(
            name=unique_name(_SEED_PASSTHROUGH_PROCESSOR_BASENAME, existing_names),
            column_names=[],
        )
    )


def _explicitly_dropped_configured_columns(
    *,
    columns: list[ColumnConfigT],
    processors: list[ProcessorConfigT],
) -> set[str]:
    configured_names = [column.name for column in columns]
    explicitly_dropped: set[str] = set()
    for processor in processors:
        if processor.processor_type != ProcessorType.DROP_COLUMNS:
            continue
        for pattern in processor.column_names:
            explicitly_dropped.update(name for name in configured_names if fnmatch(name, pattern))
    return explicitly_dropped


def _validate_preserved_sampler_dependencies(
    config: DataDesignerConfig,
    generated_columns: tuple[ColumnConfigT, ...],
) -> None:
    generated_outputs = {
        output_name for column in generated_columns for output_name in (column.name, *column.side_effect_columns)
    }
    invalid_dependencies: dict[str, list[str]] = {}

    for column in config.columns:
        if column.column_type != DataDesignerColumnType.SAMPLER:
            continue
        dependencies = _sampler_dependency_names(column)
        invalid = sorted(dependencies & generated_outputs)
        if invalid:
            invalid_dependencies[column.name] = invalid

    for constraint in config.constraints or []:
        if not isinstance(constraint, ColumnInequalityConstraint):
            continue
        if constraint.rhs in generated_outputs:
            invalid_dependencies.setdefault(constraint.target_column, []).append(constraint.rhs)

    if invalid_dependencies:
        details = ", ".join(
            f"{sampler!r} -> {sorted(set(dependencies))!r}"
            for sampler, dependencies in sorted(invalid_dependencies.items())
        )
        raise DataDesignerWorkflowError(
            "sampler_retry_mode='preserve' cannot materialize sampler columns that depend on regenerated "
            f"columns: {details}. Use sampler_retry_mode='resample' or move the dependency into the sampler cohort."
        )


def _sampler_dependency_names(column: ColumnConfigT) -> set[str]:
    dependencies = set(column.required_columns)
    for condition in getattr(column, "conditional_params", {}):
        if condition == "...":
            continue
        try:
            dependencies.update(extract_column_names_from_expression(condition))
        except Exception as exc:
            raise DataDesignerWorkflowError(
                f"Cannot inspect conditional sampler expression {condition!r} on column {column.name!r}: {exc}"
            ) from exc

    params = [getattr(column, "params", None), *getattr(column, "conditional_params", {}).values()]
    for params_config in params:
        for field_name in _SAMPLER_REFERENCE_PARAM_FIELDS:
            reference = getattr(params_config, field_name, None)
            if isinstance(reference, str):
                dependencies.add(reference)
    return dependencies
