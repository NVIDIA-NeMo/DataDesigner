# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.seed_source import SeedSource


class DemoSeedSource(SeedSource):
    seed_type: Literal["demo-seed-reader"] = "demo-seed-reader"

    directory: str
    filename: str
