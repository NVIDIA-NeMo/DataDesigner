# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data_designer.config.column_configs import CustomColumnConfig, ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.models import ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.interface.data_designer import DataDesigner


@custom_column_generator(required_columns=["value"])
def flag_missing_value(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "flag": "MISSING" if row["value"] is None else "PRESENT"}


@pytest.mark.parametrize("run", ["preview", "create"])
def test_a_missing_seed_text_value_reaches_templates_and_generators_as_none(
    tmp_path: Path, stub_model_providers: list[ModelProvider], run: str
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=managed,
    )
    designer.set_run_config(RunConfig(otel_metrics_port=None))
    config = DataDesignerConfigBuilder().with_seed_dataset(
        DataFrameSeedSource(df=pd.DataFrame({"value": ["present", None, ""]}))
    )
    config.add_column(
        ExpressionColumnConfig(name="rendered", expr="{{ 'MISSING' if value is none else 'PRESENT' }}", dtype="str")
    )
    config.add_column(CustomColumnConfig(name="flag", generator_function=flag_missing_value))

    if run == "preview":
        dataset = designer.preview(config, num_records=3).dataset
    else:
        dataset = designer.create(config, num_records=3, dataset_name="missing_values").load_dataset()

    assert dataset["rendered"].tolist() == ["PRESENT", "MISSING", "PRESENT"]
    assert dataset["flag"].tolist() == ["PRESENT", "MISSING", "PRESENT"]
