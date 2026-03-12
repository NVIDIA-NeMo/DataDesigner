# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
import sys

from data_designer.cli.ui import print_warning
from data_designer.config.default_model_settings import resolve_seed_default_model_settings

_BOOTSTRAP_COMMAND = (
    f'{shlex.quote(sys.executable)} -c "from data_designer.config.default_model_settings import '
    'resolve_seed_default_model_settings; resolve_seed_default_model_settings()"'
)

_default_model_settings_checked = False


def ensure_cli_default_model_settings() -> None:
    """Best-effort bootstrap for CLI default model settings."""
    global _default_model_settings_checked
    if _default_model_settings_checked:
        return

    try:
        resolve_seed_default_model_settings()
    except Exception as e:
        print_warning(
            "Could not initialize default model providers and model configs automatically. "
            f"The command will continue. Error: {e}. "
            f"You can retry setup with `{_BOOTSTRAP_COMMAND}` "
            "or configure providers/models manually with `data-designer config providers` "
            "and `data-designer config models`."
        )
    finally:
        _default_model_settings_checked = True
