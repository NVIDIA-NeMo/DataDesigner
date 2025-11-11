# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
import tempfile

import pandas as pd

from data_designer.config.processors import ToJsonlProcessorConfig
from data_designer.engine.configurable_task import ConfigurableTaskMetadata
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.utils import deserialize_json_values

logger = logging.getLogger(__name__)


class ToJsonlProcessor(WithJinja2UserTemplateRendering, Processor[ToJsonlProcessorConfig]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="to_jsonl",
            description="Save formatted dataset as JSONL files.",
            required_resources=None,
        )

    @property
    def template_as_string(self) -> str:
        return json.dumps(self.config.template)

    def _get_stop_index_per_file(self, dataset_size: int) -> dict[str, int]:
        """Helper function to get the end index for each file of the split."""
        stop_index_per_file = {}

        accumulated_fraction = 0.0
        for filename, fraction in self.config.fraction_per_file.items():
            accumulated_fraction += fraction
            stop_index_per_file[filename] = min(int(accumulated_fraction * dataset_size), dataset_size)

        return stop_index_per_file

    def process(self, data: pd.DataFrame, *, current_batch_number: int | None = None) -> pd.DataFrame:
        self.prepare_jinja2_template_renderer(self.template_as_string, data.columns.to_list())

        stop_index_per_file = self._get_stop_index_per_file(len(data))
        with tempfile.TemporaryDirectory() as temp_dir:
            start_index = 0
            for filename, stop_index in stop_index_per_file.items():
                logger.info(f"✏️ Writing {stop_index - start_index} formatted JSONL entries to {filename}")

                records = data.iloc[start_index:stop_index].to_dict(orient="records")
                with open(Path(temp_dir) / f"{filename}", "a") as f:
                    for i, record in enumerate(records):
                        rendered_jsonl_entry = self.render_template(deserialize_json_values(record))
                        escaped_jsonl_entry = rendered_jsonl_entry.replace("\n", "\\n")
                        f.write(escaped_jsonl_entry)
                        if i < len(records) - 1:
                            f.write("\n")
                start_index = stop_index

                self.artifact_storage.move_to_outputs(Path(temp_dir) / filename, self.config.folder_name)

        return data
