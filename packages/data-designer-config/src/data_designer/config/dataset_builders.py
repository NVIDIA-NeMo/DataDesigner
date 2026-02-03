# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class BuildStage(str, Enum):
    PRE_BATCH = "pre_batch"
    POST_BATCH = "post_batch"
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
