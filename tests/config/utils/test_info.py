# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.info import ConfigBuilderInfo, InfoType, InterfaceInfo
from data_designer.config.utils.type_helpers import get_sampler_params


@patch("data_designer.config.utils.info.display_sampler_table")
@patch("data_designer.config.utils.info.display_model_configs_table")
def test_config_builder_sampler_info(mock_display_model_configs_table, mock_display_sampler_table, stub_model_configs):
    info = ConfigBuilderInfo(model_configs=stub_model_configs)
    info.display(InfoType.MODEL_CONFIGS)
    mock_display_model_configs_table.assert_called_once_with(stub_model_configs)

    sampler_params = get_sampler_params()
    info.display(InfoType.SAMPLERS)
    mock_display_sampler_table.assert_called_once_with(sampler_params)

    mock_display_sampler_table.reset_mock()
    info.display(InfoType.SAMPLERS, sampler_type=SamplerType.BERNOULLI)
    mock_display_sampler_table.assert_called_once_with(
        {SamplerType.BERNOULLI: sampler_params[SamplerType.BERNOULLI]}, title="Bernoulli Sampler"
    )


@patch("data_designer.config.utils.info.display_model_configs_table")
def test_config_builder_model_configs_info(mock_display_model_configs_table, stub_model_configs):
    info = ConfigBuilderInfo(model_configs=stub_model_configs)
    info.display(InfoType.MODEL_CONFIGS)
    mock_display_model_configs_table.assert_called_once_with(stub_model_configs)


def test_config_builder_unsupported_info_type(stub_model_configs):
    info = ConfigBuilderInfo(model_configs=stub_model_configs)
    with pytest.raises(
        ValueError,
        match="Unsupported info_type: 'unsupported_type'. ConfigBuilderInfo only supports 'samplers' and 'model_configs'.",
    ):
        info.display("unsupported_type")


@patch("data_designer.config.utils.info.display_model_providers_table")
def test_interface_model_providers_info(mock_display_model_providers_table, stub_model_providers):
    info = InterfaceInfo(model_providers=stub_model_providers)
    info.display(InfoType.MODEL_PROVIDERS)
    mock_display_model_providers_table.assert_called_once_with(stub_model_providers)


def test_interface_unsupported_info_type(stub_model_providers):
    info = InterfaceInfo(model_providers=stub_model_providers)
    with pytest.raises(
        ValueError, match="Unsupported info_type: 'unsupported_type'. InterfaceInfo only supports 'model_providers'."
    ):
        info.display("unsupported_type")
