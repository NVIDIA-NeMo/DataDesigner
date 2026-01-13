# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.column_configs import SingleColumnConfig


class DemoColumnGeneratorConfig(SingleColumnConfig):
    column_type: Literal["demo-column-generator"] = "demo-column-generator"

    text: str
