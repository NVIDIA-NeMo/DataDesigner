# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataDesignerScriptParams:
    """Runtime parameters forwarded to Python config workflows.

    Attributes:
        argv: Raw workflow arguments passed after the CLI ``--`` separator.
    """

    argv: tuple[str, ...] = ()
