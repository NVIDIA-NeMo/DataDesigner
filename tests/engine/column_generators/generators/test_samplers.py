# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.sampler_params import (
    BernoulliMixtureSamplerParams,
    BernoulliSamplerParams,
    BinomialSamplerParams,
    CategorySamplerParams,
    DatetimeSamplerParams,
    GaussianSamplerParams,
    PersonFromFakerSamplerParams,
    PoissonSamplerParams,
    SamplerType,
    ScipySamplerParams,
    SubcategorySamplerParams,
    TimeDeltaSamplerParams,
    UniformSamplerParams,
    UUIDSamplerParams,
)
from data_designer.engine.column_generators.generators.samplers import SamplerColumnGenerator
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig


@pytest.fixture
def stub_sampler_column_configs():
    return [
        SamplerColumnConfig(
            name="person",
            sampler_type=SamplerType.PERSON_FROM_FAKER,
            params=PersonFromFakerSamplerParams(
                # non en_US uses Faker
                locale="en_GB",
                age_range=[25, 70],
            ),
        ),
        SamplerColumnConfig(
            name="category",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["cat_1", "cat_2", "cat_3"],
                weights=[0.1, 0.2, 0.7],
            ),
        ),
        SamplerColumnConfig(
            name="subcategory",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="category",
                values={
                    "cat_1": ["subcat_1", "subcat_2"],
                    "cat_2": ["subcat_3", "subcat_4"],
                    "cat_3": ["subcat_5", "subcat_6"],
                },
            ),
        ),
        SamplerColumnConfig(
            name="gaussian",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=0.5, stddev=0.1, decimal_places=2),
        ),
        SamplerColumnConfig(
            name="datetime",
            sampler_type=SamplerType.DATETIME,
            params=DatetimeSamplerParams(start="2020-01-01", end="2025-01-01"),
            convert_to="%m/%d/%Y",
        ),
        SamplerColumnConfig(
            name="timedelta",
            sampler_type=SamplerType.TIMEDELTA,
            params=TimeDeltaSamplerParams(dt_min=0, dt_max=100, reference_column_name="datetime"),
        ),
        SamplerColumnConfig(
            name="uuid",
            sampler_type=SamplerType.UUID,
            params=UUIDSamplerParams(prefix="test_", short_form=True, uppercase=True),
        ),
        SamplerColumnConfig(
            name="binomial",
            sampler_type=SamplerType.BINOMIAL,
            params=BinomialSamplerParams(n=100, p=0.5),
        ),
        SamplerColumnConfig(
            name="bernoulli",
            sampler_type=SamplerType.BERNOULLI,
            params=BernoulliSamplerParams(p=0.5),
        ),
        SamplerColumnConfig(
            name="bernoulli_mixture",
            sampler_type=SamplerType.BERNOULLI_MIXTURE,
            params=BernoulliMixtureSamplerParams(p=0.5, dist_name="binom", dist_params={"n": 100, "p": 0.5}),
        ),
        SamplerColumnConfig(
            name="poisson",
            sampler_type=SamplerType.POISSON,
            params=PoissonSamplerParams(mean=1.0),
        ),
        SamplerColumnConfig(
            name="uniform",
            sampler_type=SamplerType.UNIFORM,
            params=UniformSamplerParams(low=0.0, high=1.0, decimal_places=2),
        ),
        SamplerColumnConfig(
            name="scipy",
            sampler_type=SamplerType.SCIPY,
            params=ScipySamplerParams(dist_name="norm", dist_params={"loc": 0.0, "scale": 1.0}, decimal_places=2),
        ),
    ]


def test_sampler_generator_generate_from_scratch(
    stub_sampler_column_configs,
    stub_resource_provider,
):
    generator = SamplerColumnGenerator(
        config=SamplerMultiColumnConfig(columns=stub_sampler_column_configs),
        resource_provider=stub_resource_provider,
    )
    df = generator.generate_from_scratch(num_records=100)
    assert df.shape == (100, len(stub_sampler_column_configs))


def test_sampler_generator_generate(stub_dataframe, stub_sampler_column_configs, stub_resource_provider, tmp_path):
    generator = SamplerColumnGenerator(
        config=SamplerMultiColumnConfig(columns=stub_sampler_column_configs),
        resource_provider=stub_resource_provider,
    )
    df = generator.generate(stub_dataframe)
    assert df.shape == (len(stub_dataframe), len(stub_sampler_column_configs) + len(stub_dataframe.columns))
