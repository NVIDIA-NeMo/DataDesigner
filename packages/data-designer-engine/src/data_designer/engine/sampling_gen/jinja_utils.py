# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from jinja2 import meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.run_config import JinjaRenderingEngine
from data_designer.engine.processing.ginja.environment import (
    WithJinja2UserTemplateRendering,
)

if TYPE_CHECKING:
    import pandas as pd


class JinjaDataFrame(WithJinja2UserTemplateRendering):
    def __init__(
        self,
        expr: str,
        *,
        jinja_rendering_engine: JinjaRenderingEngine = JinjaRenderingEngine.NATIVE,
    ):
        self.expr = expr
        self._jinja_rendering_engine = jinja_rendering_engine

    def _jsonify(self, record) -> dict[str, Any]:
        for key, value in record.items():
            if isinstance(value, lazy.pd.Timestamp):
                record[key] = value.isoformat()
        return record

    @property
    def _expr(self) -> str:
        return "{{ " + self.expr + " }}"

    def select_index(self, dataframe: pd.DataFrame) -> pd.Index:
        if dataframe.empty or self.expr == "...":
            return dataframe.index

        self.prepare_jinja2_template_renderer(self._expr, list(dataframe))

        where = dataframe.apply(
            lambda row: self.render_template(self._jsonify(row.to_dict())) == "True",
            axis=1,
        ).to_numpy()

        return dataframe[where].index

    def to_column(self, dataframe: pd.DataFrame) -> list[Any]:
        self.prepare_jinja2_template_renderer(self._expr, list(dataframe))

        expr_values = []
        for record in dataframe.to_dict(orient="records"):
            rendered = self.render_template(self._jsonify(record))
            try:
                # Non-string expressions are evaluated as literals.
                expr_values.append(ast.literal_eval(rendered))
            except (ValueError, SyntaxError):
                # Strings throw an error and are appended directly.
                expr_values.append(rendered)

        return expr_values


def extract_column_names_from_expression(expr: str) -> set[str]:
    """Extract valid column names from the given expression."""
    return meta.find_undeclared_variables(ImmutableSandboxedEnvironment().parse("{{ " + expr + " }}"))
