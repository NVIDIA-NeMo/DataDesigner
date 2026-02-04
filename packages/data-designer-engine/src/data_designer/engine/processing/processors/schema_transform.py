# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from data_designer.config.processors import SchemaTransformProcessorConfig
from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.utils import deserialize_json_values
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _json_escape_record(record: dict[str, Any]) -> dict[str, Any]:
    """Escape record values for safe insertion into a JSON template."""

    def escape_for_json_string(s: str) -> str:
        """Use json.dumps to escape, then strip the surrounding quotes."""
        return json.dumps(s)[1:-1]

    escaped = {}
    for key, value in record.items():
        if isinstance(value, str):
            escaped[key] = escape_for_json_string(value)
        elif isinstance(value, (dict, list)):
            escaped[key] = escape_for_json_string(json.dumps(value))
        elif value is None:
            escaped[key] = "null"
        else:
            escaped[key] = str(value)
    return escaped


class SchemaTransformProcessor(WithJinja2UserTemplateRendering, Processor[SchemaTransformProcessorConfig]):
    """Transforms dataset schema using Jinja2 templates after each batch."""

    @property
    def template_as_str(self) -> str:
        return json.dumps(self.config.template)

    def process_after_batch(self, data: pd.DataFrame, *, batch_number: int) -> pd.DataFrame:
        formatted_data = self._transform(data)
        self.artifact_storage.write_batch_to_parquet_file(
            batch_number=batch_number,
            dataframe=formatted_data,
            batch_stage=BatchStage.PROCESSORS_OUTPUTS,
            subfolder=self.config.name,
        )
        return data

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.prepare_jinja2_template_renderer(self.template_as_str, data.columns.to_list())
        formatted_records = []
        for record in data.to_dict(orient="records"):
            deserialized = deserialize_json_values(record)
            escaped = _json_escape_record(deserialized)
            rendered = self.render_template(escaped)
            formatted_records.append(json.loads(rendered))
        return pd.DataFrame(formatted_records)
