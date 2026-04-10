# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# IMPORTANT: This module must NOT import from any data_designer submodules (i.e., data_designer.*).

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

from jinja2 import meta as jinja2_meta
from jinja2.exceptions import TemplateSyntaxError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

# Shared env for Jinja2 AST parsing (syntax checks + variable extraction).
# Cannot reuse misc.py helpers because base.py must not import data_designer.*.
_VALIDATION_ENV = ImmutableSandboxedEnvironment()


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        json_schema_mode_override="validation",
    )


class SkipConfig(ConfigBase):
    """Expression gate for conditional column generation.

    Attach to a ``SingleColumnConfig`` via ``skip=SkipConfig(...)`` to gate
    generation on a Jinja2 expression.  Controls *when* to skip; propagation
    of upstream skips is controlled separately by ``propagate_skip`` on
    ``SingleColumnConfig``.

    Attributes:
        when: Jinja2 expression (including ``{{ }}`` delimiters); when truthy,
            skip generation for this row.
        value: Value to write for skipped cells.  Defaults to ``None``
            (becomes ``NaN``/``pd.NA`` in the DataFrame).
    """

    when: str = Field(
        description="Jinja2 expression (including {{ }} delimiters); when truthy, skip generation for this row.",
    )
    value: bool | int | float | str | None = Field(
        default=None,
        description="Value to write for skipped cells. Defaults to None (becomes NaN/pd.NA in DataFrame).",
    )

    @field_validator("when")
    @classmethod
    def _validate_when_syntax(cls, v: str) -> str:
        try:
            ast = _VALIDATION_ENV.parse(v)
        except TemplateSyntaxError as exc:
            raise ValueError(str(exc)) from exc
        if not jinja2_meta.find_undeclared_variables(ast):
            raise ValueError(
                f"skip.when expression {v!r} does not reference any columns. "
                "Expressions must use Jinja2 delimiters, e.g. "
                "'{{ in_stock == 0 }}' not 'in_stock == 0'."
            )
        return v

    # cached_property writes to instance.__dict__; this works because ConfigBase
    # is not frozen.  If ConfigBase ever gains frozen=True, switch to model_post_init.
    @cached_property
    def columns(self) -> list[str]:
        """Column names referenced in the ``when`` expression.

        Parsed once from the Jinja2 AST and cached.  Used by the DAG builder
        to add dependency edges and by the execution graph to store metadata.
        """
        ast = _VALIDATION_ENV.parse(self.when)
        return sorted(jinja2_meta.find_undeclared_variables(ast))


class SingleColumnConfig(ConfigBase, ABC):
    """Abstract base class for all single-column configuration types.

    This class serves as the foundation for all column configurations in DataDesigner,
    defining shared fields and properties across all column type.

    Attributes:
        name: Unique name of the column to be generated.
        drop: If True, the column will be generated but removed from the final dataset.
            Useful for intermediate columns that are dependencies for other columns.
        column_type: Discriminator field that identifies the specific column type.
            Subclasses must override this field to specify the column type with a `Literal` value.
        skip: Optional expression gate for conditional generation.
        propagate_skip: If True (default), this column auto-skips when any of its
            required_columns was skipped.  Independent of ``skip``.
    """

    name: str
    drop: bool = False
    allow_resize: bool = False
    column_type: str
    skip: SkipConfig | None = None
    propagate_skip: bool = Field(
        default=True,
        description="If True (default), this column auto-skips when any "
        "of its required_columns was skipped. Independent of skip — "
        "a column with no SkipConfig still propagates upstream skips. "
        "Set to False for null-tolerant columns.",
    )

    @model_validator(mode="after")
    def _validate_skip_scope(self) -> Self:
        if self.skip is not None:
            if self.column_type in ("sampler", "seed-dataset"):
                raise ValueError(
                    f"skip is not supported on {self.column_type} columns. "
                    "Sampler/seed columns are collapsed into shared multi-column generators "
                    "and cannot be skipped individually."
                )
            if self.allow_resize:
                raise ValueError(
                    "skip and allow_resize cannot be used together. "
                    "allow_resize changes buffer size during generation (1:N / N:1), which "
                    "breaks index-based skip tracking and merge-back."
                )
            self_refs = {self.name} | set(self.side_effect_columns)
            if not self_refs.isdisjoint(self.skip.columns):
                offending = self_refs & set(self.skip.columns)
                raise ValueError(
                    f"skip.when expression for column '{self.name}' references itself "
                    f"(via {offending!r}). A column cannot gate its own generation."
                )
        return self

    @staticmethod
    def get_column_emoji() -> str:
        return "🎨"

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Returns a list of column names that must exist before this column can be generated.

        Returns:
            List of column names that this column depends on. Empty list indicates
            no dependencies. Override in subclasses to specify dependencies.
        """

    @property
    @abstractmethod
    def side_effect_columns(self) -> list[str]:
        """Returns a list of additional columns that this column will create as a side effect.

        Some column types generate additional metadata or auxiliary columns alongside
        the primary column (e.g., reasoning traces for LLM columns).

        Returns:
            List of column names that this column will create as a side effect. Empty list
            indicates no side effect columns. Override in subclasses to specify side effects.
        """


class ProcessorConfig(ConfigBase, ABC):
    """Abstract base class for all processor configuration types.

    Processors are transformations that run at different stages of the generation
    pipeline. They can modify, reshape, or augment the dataset.

    Attributes:
        name: Unique name of the processor, used to identify the processor in results
            and to name output artifacts on disk.
    """

    name: str = Field(
        description="The name of the processor, used to identify the processor in the results and to write the artifacts to disk.",
    )
    processor_type: str
