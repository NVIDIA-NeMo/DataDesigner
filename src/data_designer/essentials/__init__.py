# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.exports import *
from data_designer.config.validator_params import LocalCallableValidatorParams
from data_designer.interface.data_designer import DataDesigner
from data_designer.logging import LoggingConfig, configure_logging

configure_logging(LoggingConfig.default())


def get_essentials_exports() -> list[str]:
    logging = [
        configure_logging.__name__,
        LoggingConfig.__name__,
    ]
    local = [
        DataDesigner.__name__,
        LocalCallableValidatorParams.__name__,
    ]

    return logging + local + get_config_exports()


__all__ = get_essentials_exports()
