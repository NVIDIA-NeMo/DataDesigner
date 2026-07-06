# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn
from data_designer.engine.column_generators.utils.errors import ExpressionTemplateRenderError
from data_designer.engine.context import format_row_group_tag
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.ginja.exceptions import EmptyTemplateRenderError, UserTemplateError
from data_designer.engine.processing.utils import deserialize_json_values

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

EMPTY_RENDERED_EXPRESSION = "EmptyRenderedExpression"
TEMPLATE_RENDER_ERROR = "TemplateRenderError"
TYPE_CAST_ERROR = "TypeCastError"
_VALID_DTYPES = {"str", "float", "int", "bool"}


class ExpressionColumnGenerator(WithJinja2UserTemplateRendering, ColumnGeneratorFullColumn[ExpressionColumnConfig]):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"🧩 {format_row_group_tag()}Generating column `{self.config.name}` from expression")

        missing_columns = list(set(self.config.required_columns) - set(data.columns))
        if len(missing_columns) > 0:
            error_msg = (
                f"There was an error preparing the Jinja2 expression template. "
                f"The following columns {missing_columns} are missing!"
            )
            raise ExpressionTemplateRenderError(error_msg)

        if self.config.dtype not in _VALID_DTYPES:
            raise ValueError(f"Invalid dtype: {self.config.dtype}")

        self.prepare_jinja2_template_renderer(self.config.expr, data.columns.to_list())
        records: list[dict] = []
        retained_indexes: list[object] = []
        drop_counts: Counter[str] = Counter()

        # ``DataFrame.to_dict(orient="records")`` returns ``[]`` for a 0-column
        # DataFrame regardless of its row count, so a root expression dispatched
        # against a fresh row group buffer (no upstream columns) would otherwise
        # skip rendering entirely. Synthesize empty per-row dicts in that case so
        # the loop still fires once per row and the template can render its
        # constant value (or drop the row uniformly).
        row_indexes = data.index.to_list()
        row_records = data.to_dict(orient="records")
        if not row_records and row_indexes:
            row_records = [{} for _ in row_indexes]

        for row_index, record in zip(row_indexes, row_records, strict=True):
            prepared_record = deserialize_json_values(record)
            try:
                rendered_value = self.render_template(prepared_record)
            except EmptyTemplateRenderError:
                drop_counts[EMPTY_RENDERED_EXPRESSION] += 1
                continue
            except Exception:
                logger.debug(
                    "Expression column %r dropped row %r after template render failure.",
                    self.config.name,
                    row_index,
                    exc_info=True,
                )
                drop_counts[TEMPLATE_RENDER_ERROR] += 1
                continue

            if self._is_empty_rendered_expression(rendered_value):
                drop_counts[EMPTY_RENDERED_EXPRESSION] += 1
                continue

            try:
                record[self.config.name] = self._cast_type(rendered_value)
            except (OverflowError, TypeError, ValueError):
                drop_counts[TYPE_CAST_ERROR] += 1
                continue

            records.append(record)
            retained_indexes.append(row_index)

        total_dropped = sum(drop_counts.values())
        if total_dropped > 0:
            self._log_row_drops(drop_counts, input_count=len(data), retained_count=len(records))
            if len(records) == 0:
                raise UserTemplateError(f"Expression column {self.config.name!r} produced no valid rows.")

        return lazy.pd.DataFrame(records, index=retained_indexes)

    @staticmethod
    def _is_empty_rendered_expression(value: object) -> bool:
        if value is None:
            return True
        return isinstance(value, str) and len(value.strip()) == 0

    def _log_row_drops(self, drop_counts: Counter[str], *, input_count: int, retained_count: int) -> None:
        breakdown = ", ".join(f"{name}={count}" for name, count in sorted(drop_counts.items()))
        total_dropped = sum(drop_counts.values())
        message = (
            f"Expression column {self.config.name!r} dropped {total_dropped}/{input_count} rows after render: "
            f"{breakdown}."
        )
        if retained_count == 0:
            logger.error(message)
        else:
            logger.warning(f"{message} Continuing with {retained_count} rows.")

    def _cast_type(self, value: str) -> str | float | int | bool:
        if self.config.dtype == "str":
            return value
        elif self.config.dtype == "float":
            return float(value)
        elif self.config.dtype == "int":
            return int(float(value))
        elif self.config.dtype == "bool":
            try:
                return bool(int(float(value)))
            except ValueError:
                return bool(f"{value}".lower() == "true")
        else:
            raise ValueError(f"Invalid dtype: {self.config.dtype}")
