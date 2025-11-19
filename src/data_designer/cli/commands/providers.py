# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.cli.controllers.provider_controller import ProviderController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def providers_command() -> None:
    """Configure model providers interactively."""
    controller = ProviderController(DATA_DESIGNER_HOME)
    controller.run()
