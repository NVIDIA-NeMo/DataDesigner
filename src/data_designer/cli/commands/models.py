# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.cli.controllers.model_controller import ModelController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def models_command() -> None:
    controller = ModelController(DATA_DESIGNER_HOME)
    controller.run()
