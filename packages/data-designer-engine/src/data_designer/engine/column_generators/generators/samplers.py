# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from functools import partial
from typing import TYPE_CHECKING, Callable

from data_designer.config.utils.constants import LOCALES_WITH_MANAGED_DATASETS
from data_designer.engine.column_generators.generators.base import FromScratchColumnGenerator, GenerationStrategy
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
from data_designer.engine.processing.utils import concat_datasets
from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
from data_designer.engine.sampling_gen.data_sources.sources import SamplerType
from data_designer.engine.sampling_gen.entities.person import load_person_data_sampler
from data_designer.engine.sampling_gen.generator import DatasetGenerator as SamplingDatasetGenerator
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SamplerColumnGenerator(FromScratchColumnGenerator[SamplerMultiColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        df_samplers = self.generate_from_scratch(len(data))
        return concat_datasets([data, df_samplers])

    def generate_from_scratch(self, num_records: int) -> pd.DataFrame:
        sampling_generator = self._prepare_for_generation(num_records)
        return sampling_generator.generate(num_records)

    @property
    def _needs_person_generator(self) -> bool:
        columns = [c for c in self.config.columns if c.sampler_type == SamplerType.PERSON]
        return any(c.params.locale in LOCALES_WITH_MANAGED_DATASETS for c in columns)

    @property
    def _person_generator_loader(self) -> Callable[[bool], ManagedDatasetGenerator]:
        return partial(load_person_data_sampler, blob_storage=self.resource_provider.blob_storage)

    def _create_sampling_dataset_generator(self) -> SamplingDatasetGenerator:
        return SamplingDatasetGenerator(
            sampler_columns=self.config,
            person_generator_loader=(self._person_generator_loader if self._needs_person_generator else None),
        )

    def _log_person_generation_if_needed(self) -> None:
        if self._needs_person_generator:
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
