# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random

import pandas as pd

from data_designer.engine.column_generators.generators.base import (
    FromScratchColumnGenerator,
    GenerationStrategy,
    GeneratorMetadata,
)
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
from data_designer.engine.processing.utils import concat_datasets
from data_designer.engine.sampling_gen.data_sources.sources import SamplerType
from data_designer.engine.sampling_gen.generator import DatasetGenerator as SamplingDatasetGenerator

logger = logging.getLogger(__name__)


class SamplerColumnGenerator(FromScratchColumnGenerator[SamplerMultiColumnConfig]):
    @staticmethod
    def metadata() -> GeneratorMetadata:
        return GeneratorMetadata(
            name="sampler_column_generator",
            description="Generate columns using sampling-based method.",
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            required_resources=None,
        )

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        df_samplers = self.generate_from_scratch(len(data))
        return concat_datasets([data, df_samplers])

    def generate_from_scratch(self, num_records: int) -> pd.DataFrame:
        sampling_generator = self._prepare_for_generation(num_records)
        return sampling_generator.generate(num_records)

    def _create_sampling_dataset_generator(self) -> SamplingDatasetGenerator:
        return SamplingDatasetGenerator(
            sampler_columns=self.config,
            sampler_dataset_repository=self.resource_provider.sampler_dataset_repository,
        )

    def _has_person_samplers(self) -> bool:
        return any([c.sampler_type == SamplerType.PERSON for c in self.config.columns])

    def _log_person_generation_if_needed(self) -> None:
        if self._has_person_samplers():
            columns = [c for c in self.config.columns if c.sampler_type == SamplerType.PERSON]
            emoji = random.choice(["🧑‍🎨", "🙋‍♂️", "🙋‍♀️", "🧑‍🚀", "👩‍🎤", "👨‍🍳", "👩‍🔬", "👨‍💻", "👩‍💼"])
            log_msg = f"🎲 {emoji} Initializing person generation"
            if any(c.params.with_synthetic_personas for c in columns):
                log_msg += " ⚡️ with synthetic personas ⚡️"
            logger.info(log_msg)

    def _prepare_for_generation(self, num_records: int) -> SamplingDatasetGenerator:
        logger.info(
            f"🎲 Preparing samplers to generate {num_records} records across {len(self.config.columns)} columns"
        )
        self._log_person_generation_if_needed()
        return self._create_sampling_dataset_generator()

    def _validate(self) -> None:
        if self.resource_provider.sampler_dataset_repository is None and self._has_person_samplers():
            raise ValueError("The Dataset Manager is required to use the Person Sampler.")
