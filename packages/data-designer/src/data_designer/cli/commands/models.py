# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.utils.constants import DATA_DESIGNER_HOME

# Controllers are imported inside command functions to avoid pulling in heavy
# dependencies (engine, models) at CLI startup time.


def models_command() -> None:
    from data_designer.cli.controllers.model_controller import ModelController

    controller = ModelController(DATA_DESIGNER_HOME)
    controller.run()
