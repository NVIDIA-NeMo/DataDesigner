# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.ui import print_warning
from data_designer.config.default_model_settings import resolve_seed_default_model_settings


def ensure_cli_default_model_settings() -> None:
    """Best-effort bootstrap for CLI default model settings."""
    try:
        resolve_seed_default_model_settings()
    except Exception as e:
        print_warning(
            "Could not initialize default model providers and model configs automatically. "
            f"The command will continue. Error: {e}. "
            "You will need to configure providers and models manually with "
            "`data-designer config providers` and `data-designer config models`."
        )
