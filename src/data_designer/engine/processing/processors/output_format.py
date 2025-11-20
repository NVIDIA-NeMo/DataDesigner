# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from data_designer.config.processors import OutputFormatProcessorConfig
from data_designer.engine.configurable_task import ConfigurableTaskMetadata
from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.utils import deserialize_json_values

logger = logging.getLogger(__name__)


class OutputFormatProcessor(WithJinja2UserTemplateRendering, Processor[OutputFormatProcessorConfig]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="output_format",
            description="Format the dataset using a Jinja2 template.",
            required_resources=None,
        )

    def process(self, data: pd.DataFrame, *, current_batch_number: int | None = None) -> pd.DataFrame:
        self.prepare_jinja2_template_renderer(self.config.template, data.columns.to_list())
        formatted_records = [
            self.render_template(deserialize_json_values(record)) for record in data.to_dict(orient="records")
        ]
        formatted_data = pd.DataFrame(formatted_records, columns=["formatted_output"])
        if current_batch_number is not None:
            self.artifact_storage.write_batch_to_parquet_file(
                batch_number=current_batch_number,
                dataframe=formatted_data,
                batch_stage=BatchStage.PROCESSORS_OUTPUTS,
                subfolder=self.config.name,
            )
        else:
            logger.warning("⚠️ Cannot write processor outputs to disk in preview mode.")

        return data
