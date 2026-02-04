# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from data_designer.config.column_configs import ImageGenerationColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel, GenerationStrategy
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.utils import deserialize_json_values


class ImageCellGenerator(WithJinja2UserTemplateRendering, ColumnGeneratorWithModel[ImageGenerationColumnConfig]):
    """Generator for image columns using either autoregressive or diffusion models.

    Automatically detects the appropriate API based on the model's inference parameters:
    - ChatCompletionImageGenerationInferenceParams → Responses API (GPT-5, gpt-image-*, Gemini)
    - DiffusionImageGenerationInferenceParams → image_generation API (DALL-E, Imagen, Stable Diffusion)
    """

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

        # Generate image (automatically routes to appropriate API based on inference params)
        # Returns base64-encoded image data or URL depending on configuration
        image_data = self.model.generate_image(prompt=prompt)
        data[self.config.name] = image_data
        return data
