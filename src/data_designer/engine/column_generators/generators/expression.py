# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn
from data_designer.engine.column_generators.utils.errors import ExpressionTemplateRenderError
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.utils import deserialize_json_values
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ExpressionColumnGenerator(WithJinja2UserTemplateRendering, ColumnGeneratorFullColumn[ExpressionColumnConfig]):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"🧩 Generating column `{self.config.name}` from expression")

        missing_columns = list(set(self.config.required_columns) - set(data.columns))
        if len(missing_columns) > 0:
            error_msg = (
                f"There was an error preparing the Jinja2 expression template. "
                f"The following columns {missing_columns} are missing!"
            )
            raise ExpressionTemplateRenderError(error_msg)

        self.prepare_jinja2_template_renderer(self.config.expr, data.columns.to_list())
        records = []
        for record in data.to_dict(orient="records"):
            record[self.config.name] = self._cast_type(self.render_template(deserialize_json_values(record)))
            records.append(record)

        return pd.DataFrame(records)

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
