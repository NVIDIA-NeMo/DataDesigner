# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.default_model_settings import resolve_seed_default_model_settings
from data_designer.config.exports import *  # noqa: F403
from data_designer.config.validator_params import LocalCallableValidatorParams
from data_designer.interface.data_designer import DataDesigner
from data_designer.logging import LoggingConfig, configure_logging

configure_logging(LoggingConfig.default())

# Resolve default model settings on import to ensure they are available when the library is used.
resolve_seed_default_model_settings()


def get_essentials_exports() -> list[str]:
    logging = [
        configure_logging.__name__,
        LoggingConfig.__name__,
    ]
    local = [
        DataDesigner.__name__,
        LocalCallableValidatorParams.__name__,
    ]

    return logging + local + get_config_exports()  # noqa: F405


__all__ = get_essentials_exports()
