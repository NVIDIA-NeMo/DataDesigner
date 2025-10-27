# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from data_designer.config.sampler_params import BernoulliSamplerParams, BinomialSamplerParams, SamplerType
from data_designer.config.utils.info import DataDesignerInfo


@patch("data_designer.config.utils.info.display_sampler_table")
@patch("data_designer.config.utils.info.get_sampler_params")
def test_data_designer_info(mock_get_sampler_params, mock_display_sampler_table):
    stub_bernoulli_params = BernoulliSamplerParams(p=0.5)
    stub_binomial_params = BinomialSamplerParams(n=100, p=0.5)
    mock_get_sampler_params.return_value = {
        SamplerType.BERNOULLI: stub_bernoulli_params,
        SamplerType.BINOMIAL: stub_binomial_params,
    }
    info = DataDesignerInfo()

    assert SamplerType.BINOMIAL.value in info.sampler_types
    mock_get_sampler_params.assert_called_once()

    _ = info.sampler_table
    mock_display_sampler_table.assert_called_once_with(
        {SamplerType.BERNOULLI: stub_bernoulli_params, SamplerType.BINOMIAL: stub_binomial_params}
    )

    mock_display_sampler_table.reset_mock()
    info.display_sampler(SamplerType.BERNOULLI)
    mock_display_sampler_table.assert_called_once_with(
        {SamplerType.BERNOULLI: stub_bernoulli_params}, title="Bernoulli Sampler"
    )
