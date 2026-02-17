# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from data_designer.config.seed_source import SeedSource


class DataFrameSeedSource(SeedSource):
    seed_type: Literal["df"] = "df"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: SkipJsonSchema[pd.DataFrame] = Field(
        ...,
        exclude=True,
        description=(
            "DataFrame to use directly as the seed dataset. NOTE: if you need to write a Data Designer config, "
            "you must use `LocalFileSeedSource` instead, since DataFrame objects are not serializable."
        ),
    )
