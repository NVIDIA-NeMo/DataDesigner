# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.config.column_configs import ImageGenerationColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel, GenerationStrategy
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.utils import deserialize_json_values

if TYPE_CHECKING:
    from data_designer.engine.storage.multimedia_storage import MultimediaStorage


class ImageCellGenerator(WithJinja2UserTemplateRendering, ColumnGeneratorWithModel[ImageGenerationColumnConfig]):
    """Generator for image columns with optional disk persistence.

    Behavior depends on whether multimedia storage is available via ResourceProvider:
    - If available (create mode): Saves images to disk and stores relative paths in dataframe
    - If None (preview mode): Stores base64 directly in dataframe

    API is automatically detected based on the model name:
    - Diffusion models (DALL-E, Stable Diffusion, Imagen, etc.) → image_generation API
    - All other models → chat/completions API (default)

    Storage is accessed via ResourceProvider.artifact_storage.multimedia_storage
    """

    @property
    def multimedia_storage(self) -> MultimediaStorage | None:
        """Get multimedia storage from resource provider if available."""
        return self._resource_provider.artifact_storage.multimedia_storage

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        """Generate image(s) and optionally save to disk.

        Args:
            data: Record data

        Returns:
            Record with image path(s) (create mode) or base64 data (preview mode) added
        """
        deserialized_record = deserialize_json_values(data)

        # Validate required columns
        missing_columns = list(set(self.config.required_columns) - set(data.keys()))
        if len(missing_columns) > 0:
            error_msg = (
                f"There was an error preparing the Jinja2 expression template. "
                f"The following columns {missing_columns} are missing!"
            )
            raise ValueError(error_msg)

        # Render prompt template
        self.prepare_jinja2_template_renderer(self.config.prompt, list(deserialized_record.keys()))
        prompt = self.render_template(deserialized_record)

        # Validate prompt is non-empty
        if not prompt or not prompt.strip():
            raise ValueError(f"Rendered prompt for column {self.config.name!r} is empty")

        # Generate images (returns list of base64 strings)
        base64_images = self.model.generate_image(prompt=prompt)

        # Store in dataframe based on mode
        if self.multimedia_storage:
            # Create mode: save each image to disk and store list of relative paths
            relative_paths = [self.multimedia_storage.save_base64_image(base64_image) for base64_image in base64_images]
            data[self.config.name] = relative_paths
        else:
            # Preview mode: store list of base64 strings directly
            data[self.config.name] = base64_images

        return data
