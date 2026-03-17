# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.seed_source import FileSystemSeedSource


class DemoFileSystemSeedSource(FileSystemSeedSource):
    seed_type: Literal["demo-filesystem-seed-reader"] = "demo-filesystem-seed-reader"

    prefix: str = "plugin"
