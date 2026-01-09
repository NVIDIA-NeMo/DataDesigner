# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.essentials import DataDesigner, DataDesignerConfigBuilder, ExpressionColumnConfig
from data_designer_e2e_tests.plugins.column_generator.config import TestColumnGeneratorConfig
# from data_designer_e2e_tests.plugins.seed_dataset.config import TestSeedSource

current_dir = Path(__file__).parent


def test_column_generator_plugin():
    data_designer = DataDesigner()

    config_builder = DataDesignerConfigBuilder()
    config_builder.add_column(
        ExpressionColumnConfig(
            name="upper",
            expr="HELLO WORLD",
        )
    )

    preview = data_designer.preview(config_builder)
    capitalized = set(preview.dataset["upper"].values)

    assert capitalized == {"HELLO WORLD"}


# def test_seed_dataset_plugin():
#     data_designer = DataDesigner()
#
#     config_builder = DataDesignerConfigBuilder()
#     config_builder.with_seed_dataset(
#         TestSeedSource(
#             directory=str(current_dir),
#             filename="test.csv",
#         )
#     )
#     config_builder.add_column(
#         ExpressionColumnConfig(
#             name="full_name",
#             expr="{{ first_name }} + {{ last_name }}",
#         )
#     )
#
#     preview = data_designer.preview(config_builder)
#     full_names = set(preview.dataset["full_name"].values)
#
#     assert full_names == {"John + Coltrane", "Miles + Davis", "Bill + Evans"}
