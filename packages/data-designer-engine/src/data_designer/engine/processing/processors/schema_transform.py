# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.processors import SchemaTransformProcessorConfig
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.utils import deserialize_json_values
from data_designer.engine.storage.artifact_storage import BatchStage

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _escape_value_for_json(value: Any) -> str:
    """Escape a value for safe embedding inside a JSON string.

    Unlike prompt or expression templates (which produce plain text),
    schema transform templates produce JSON. Values interpolated into
    a JSON string must be escaped - e.g. quotes and backslashes - so
    the rendered output is valid JSON. We pass this as record_str_fn
    to also enable nested dot access ({{ col.sub.field }}) on
    deserialized JSON columns.
    """
    if isinstance(value, str):
        return json.dumps(value)[1:-1]
    if isinstance(value, (dict, list)):
        return json.dumps(json.dumps(value))[1:-1]
    if value is None:
        return "null"
    return str(value)


class SchemaTransformProcessor(WithJinja2UserTemplateRendering, Processor[SchemaTransformProcessorConfig]):
    """Transforms dataset schema using Jinja2 templates after each batch."""

    @property
    def template_as_str(self) -> str:
        return json.dumps(self.config.template)

    def process_after_batch(self, data: pd.DataFrame, *, current_batch_number: int | None) -> pd.DataFrame:
        formatted_data = self._transform(data)
        if current_batch_number is not None:
            self.artifact_storage.write_batch_to_parquet_file(
                batch_number=current_batch_number,
                dataframe=formatted_data,
                batch_stage=BatchStage.PROCESSORS_OUTPUTS,
                subfolder=self.config.name,
            )
        else:
            self.artifact_storage.write_parquet_file(
                parquet_file_name=f"{self.config.name}.parquet",
                dataframe=formatted_data,
                batch_stage=BatchStage.PROCESSORS_OUTPUTS,
            )
        return data

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.prepare_jinja2_template_renderer(
            self.template_as_str, data.columns.to_list(), record_str_fn=_escape_value_for_json
        )
        formatted_records = []
        for record in data.to_dict(orient="records"):
            deserialized = deserialize_json_values(record)
            rendered = self.render_template(deserialized)
            formatted_records.append(json.loads(rendered))
        return lazy.pd.DataFrame(formatted_records)
