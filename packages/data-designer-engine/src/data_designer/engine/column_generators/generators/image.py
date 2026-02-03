# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from litellm.types.utils import ImageResponse

from data_designer.config.column_configs import ImageGenerationColumnConfig
from data_designer.config.models import ModalityDataType
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel, GenerationStrategy
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.utils import deserialize_json_values


class ImageCellGenerator(WithJinja2UserTemplateRendering, ColumnGeneratorWithModel[ImageGenerationColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        deserialized_record = deserialize_json_values(data)
        missing_columns = list(set(self.config.required_columns) - set(data.keys()))
        if len(missing_columns) > 0:
            error_msg = (
                f"There was an error preparing the Jinja2 expression template. "
                f"The following columns {missing_columns} are missing!"
            )
            raise ValueError(error_msg)

        self.prepare_jinja2_template_renderer(self.config.prompt, list(deserialized_record.keys()))
        prompt = self.render_template(deserialized_record)
        image_response: ImageResponse = self.model.generate_image(prompt=prompt)
        if self.model_config.inference_parameters.output_format == ModalityDataType.URL:
            data[self.config.name] = image_response.data[0].url
        else:
            data[self.config.name] = image_response.data[0].b64_json
        return data
