# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.seed_source import FileSystemSeedSource


class MarkdownSectionSeedSource(FileSystemSeedSource):
    seed_type: Literal["markdown-section-seed-reader"] = "markdown-section-seed-reader"
    file_pattern: str = "*.md"
