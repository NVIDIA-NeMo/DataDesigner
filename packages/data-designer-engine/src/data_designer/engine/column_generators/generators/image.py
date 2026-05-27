# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from data_designer.config.column_configs import ImageColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel, GenerationStrategy
from data_designer.engine.processing.ginja.environment import WithJinja2UserTemplateRendering
from data_designer.engine.processing.utils import deserialize_json_values

if TYPE_CHECKING:
    from data_designer.engine.storage.media_storage import MediaStorage


class ImageCellGenerator(WithJinja2UserTemplateRendering, ColumnGeneratorWithModel[ImageColumnConfig]):
    """Generator for image columns with disk or dataframe persistence.

    Media storage always exists and determines behavior via its mode:
    - DISK mode: Saves images to disk and stores relative paths in dataframe
    - DATAFRAME mode: Stores base64 directly in dataframe
    """

    @property
    def media_storage(self) -> MediaStorage:
        """Get media storage from resource provider."""
        return self._resource_provider.artifact_storage.media_storage

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def _prepare_image_inputs(self, data: dict) -> tuple[str, list[dict] | None]:
        """Validate inputs and render prompt for image generation."""
        deserialized_record = deserialize_json_values(data)
        missing_columns = list(set(self.config.required_columns) - set(data.keys()))
        if len(missing_columns) > 0:
            raise ValueError(
                f"There was an error preparing the Jinja2 expression template. "
                f"The following columns {missing_columns} are missing!"
            )
        self.prepare_jinja2_template_renderer(self.config.prompt, list(deserialized_record.keys()))
        prompt = self.render_template(deserialized_record)
        if not prompt or not prompt.strip():
            raise ValueError(f"Rendered prompt for column {self.config.name!r} is empty")
        multi_modal_context = self._build_multi_modal_context(deserialized_record)
        return prompt, multi_modal_context

    def generate(self, data: dict) -> dict:
        """Generate image(s) and optionally save to disk."""
        prompt, multi_modal_context = self._prepare_image_inputs(data)
        base64_images = self.model.generate_image(prompt=prompt, multi_modal_context=multi_modal_context)
        results = [
            self.media_storage.save_base64_image(base64_image, subfolder_name=self.config.name)
            for base64_image in base64_images
        ]
        data[self.config.name] = results
        return data

    async def agenerate(self, data: dict) -> dict:
        """Native async generate using model.agenerate_image."""
        prompt, multi_modal_context = self._prepare_image_inputs(data)
        base64_images = await self.model.agenerate_image(prompt=prompt, multi_modal_context=multi_modal_context)
        results = await asyncio.to_thread(
            lambda: [
                self.media_storage.save_base64_image(base64_image, subfolder_name=self.config.name)
                for base64_image in base64_images
            ]
        )
        data[self.config.name] = results
        return data
