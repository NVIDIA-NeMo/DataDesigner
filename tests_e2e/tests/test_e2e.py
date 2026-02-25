# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer_e2e_tests.plugins.column_generator.config import DemoColumnGeneratorConfig
from data_designer_e2e_tests.plugins.regex_filter.config import RegexFilterProcessorConfig
from data_designer_e2e_tests.plugins.seed_reader.config import DemoSeedSource


def test_column_generator_plugin() -> None:
    data_designer = DataDesigner()

    config_builder = dd.DataDesignerConfigBuilder()
    # This sampler column is necessary as a temporary workaround to https://github.com/NVIDIA-NeMo/DataDesigner/issues/4
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="irrelevant",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["irrelevant"]),
        )
    )
    config_builder.add_column(
        DemoColumnGeneratorConfig(
            name="upper",
            text="hello world",
        )
    )

    preview = data_designer.preview(config_builder)
    capitalized = set(preview.dataset["upper"].values)

    assert capitalized == {"HELLO WORLD"}


def test_seed_reader_plugin() -> None:
    current_dir = Path(__file__).parent

    data_designer = DataDesigner()

    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(
        DemoSeedSource(
            directory=str(current_dir),
            filename="test_seed.csv",
        )
    )
    # This sampler column is necessary as a temporary workaround to https://github.com/NVIDIA-NeMo/DataDesigner/issues/4
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="irrelevant",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["irrelevant"]),
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="full_name",
            expr="{{ first_name }} + {{ last_name }}",
        )
    )

    preview = data_designer.preview(config_builder)
    full_names = set(preview.dataset["full_name"].values)

    assert full_names == {"John + Coltrane", "Miles + Davis", "Bill + Evans"}


def test_processor_plugin() -> None:
    seed_data = pd.DataFrame(
        {
            "category": ["keep", "drop", "keep", "drop"],
            "value": ["a", "b", "c", "d"],
        }
    )

    data_designer = DataDesigner()

    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_data))
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="irrelevant",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["irrelevant"]),
        )
    )
    config_builder.add_processor(
        RegexFilterProcessorConfig(
            name="keep_only",
            column="category",
            pattern="^keep$",
        )
    )

    preview = data_designer.preview(config_builder)
    assert len(preview.dataset) > 0
    assert all(v == "keep" for v in preview.dataset["category"].values)
