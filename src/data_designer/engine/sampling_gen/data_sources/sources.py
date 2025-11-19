# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

import numpy as np
import pandas as pd
from scipy import stats

from data_designer.config.sampler_params import (
    BernoulliMixtureSamplerParams,
    BernoulliSamplerParams,
    BinomialSamplerParams,
    CategorySamplerParams,
    DatetimeSamplerParams,
    GaussianSamplerParams,
    PersonFromFakerSamplerParams,
    PersonSamplerParams,
    PoissonSamplerParams,
    SamplerParamsT,
    SamplerType,
    ScipySamplerParams,
    SubcategorySamplerParams,
    TimeDeltaSamplerParams,
    UniformSamplerParams,
    UUIDSamplerParams,
)
from data_designer.engine.sampling_gen.data_sources.base import (
    DataSource,
    DatetimeFormatMixin,
    NumpyArray1dT,
    PassthroughMixin,
    Sampler,
    ScipyStatsSampler,
    TypeConversionMixin,
)
from data_designer.engine.sampling_gen.data_sources.errors import (
    InvalidSamplerParamsError,
    PersonSamplerConstraintsError,
)
from data_designer.engine.sampling_gen.entities.dataset_based_person_fields import PERSONA_FIELDS, PII_FIELDS
from data_designer.engine.sampling_gen.people_gen import PeopleGen

ONE_BILLION = 10**9


class SamplerRegistry:
    _registry: dict[str, type] = {}
    _reverse_registry: dict[type, str] = {}
    _params_registry: dict[type, type] = {}

    @classmethod
    def register(cls, alias: str):
        def decorator(wrapped_class: type[DataSource[SamplerParamsT]]) -> type:
            cls._registry[alias] = wrapped_class
            cls._reverse_registry[wrapped_class] = alias
            cls._params_registry[wrapped_class.get_param_type()] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def get_sampler(cls, alias: str) -> type[DataSource[SamplerParamsT]]:
        return cls._registry[alias.lower()]

    @classmethod
    def get_sampler_for_params(cls, params_type: SamplerParamsT) -> type[DataSource[SamplerParamsT]]:
        return cls._params_registry[type(params_type)]

    @classmethod
    def get_sampler_alias_for_params(cls, params_type: SamplerParamsT) -> str:
        return cls._reverse_registry[cls._params_registry[type(params_type)]]

    @classmethod
    def is_registered(cls, alias: str) -> bool:
        return alias in cls._registry

    @classmethod
    def validate_sampler_type(
        cls, sampler_type: str | type[DataSource[SamplerParamsT]]
    ) -> type[DataSource[SamplerParamsT]]:
        if isinstance(sampler_type, str):
            if sampler_type not in cls._registry:
                raise ValueError(
                    f"Sampler type `{sampler_type}` not found in the registry. "
                    f"Available samplers: {list(cls._registry.keys())}"
                )
            sampler_type = cls.get_sampler(sampler_type)
        if not issubclass(sampler_type, DataSource):
            raise ValueError(f"Sampler type `{sampler_type}` is not a subclass of `DataSource`")
        return sampler_type


#########################################
# Data Source Subclasses
#########################################


@SamplerRegistry.register(SamplerType.SUBCATEGORY)
class SubcategorySampler(TypeConversionMixin, DataSource[SubcategorySamplerParams]):
    def get_required_column_names(self) -> tuple[str, ...]:
        return (self.params.category,)

    def inject_data_column(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        index: list[int] | None = None,
    ) -> pd.DataFrame:
        index = slice(None) if index is None else index

        if len(index) == 0:
            return dataframe

        dataframe.loc[index, column_name] = dataframe.loc[index, self.params.category].apply(
            lambda cat_value: self.rng.choice(self.params.values[cat_value])
        )

        return dataframe


#########################################
# Sampler Subclasses
#########################################


@SamplerRegistry.register(SamplerType.CATEGORY)
class CategorySampler(TypeConversionMixin, Sampler[CategorySamplerParams]):
    def sample(self, num_samples: int) -> NumpyArray1dT:
        return self.rng.choice(self.params.values, size=num_samples, p=self.params.weights)


@SamplerRegistry.register(SamplerType.DATETIME)
class DatetimeSampler(DatetimeFormatMixin, Sampler[DatetimeSamplerParams]):
    def sample(self, num_samples: int) -> NumpyArray1dT:
        # Convert nanoseconds to seconds.
        start_sec = pd.to_datetime(self.params.start).value // ONE_BILLION
        end_sec = pd.to_datetime(self.params.end).value // ONE_BILLION

        random_ns = (ONE_BILLION * self.rng.randint(start_sec, end_sec, num_samples, dtype=np.int64)).view(
            "datetime64[ns]"
        )

        return np.array(random_ns, dtype=f"datetime64[{self.params.unit}]")


@SamplerRegistry.register(SamplerType.PERSON)
class PersonSampler(PassthroughMixin, Sampler[PersonSamplerParams]):
    def _setup(self, **kwargs) -> None:
        self._generator = None
        self._fixed_kwargs = {}
        for field in self.params.generator_kwargs:
            if getattr(self.params, field) is not None:
                attr = getattr(self.params, field)
                if field == "select_field_values":
                    for key, value in attr.items():
                        if key == "state" and self.params.locale == "en_US":
                            key = "region"  # This is the field name in the census-based person dataset.
                        if key not in PII_FIELDS + PERSONA_FIELDS:
                            raise ValueError(f"Invalid field name: {key}")
                        self._fixed_kwargs[key] = value
                else:
                    self._fixed_kwargs[field] = attr
        if people_gen_resource := kwargs.get("people_gen_resource"):
            if self.params.people_gen_key not in people_gen_resource:
                raise ValueError(f"Person generator with key {self.params.people_gen_key} not found.")
            self.set_generator(people_gen_resource[self.params.people_gen_key])

    def set_generator(self, generator: PeopleGen) -> None:
        self._generator = generator

    def sample(self, num_samples: int) -> NumpyArray1dT:
        if self._generator is None:
            raise ValueError("Generator not set. Please setup generator before sampling.")

        samples = np.array(self._generator.generate(num_samples, **self._fixed_kwargs))
        if len(samples) < num_samples:
            raise PersonSamplerConstraintsError(
                f"🛑 Only {len(samples)} samples could be generated with the given settings: {self._fixed_kwargs!r}. "
                "This is likely because the filter values are too strict. Person sampling does not support "
                "rare combinations of field values. Please loosen the constraints and try again."
            )
        return samples


@SamplerRegistry.register(SamplerType.PERSON_FROM_FAKER)
class PersonFromFakerSampler(PassthroughMixin, Sampler[PersonFromFakerSamplerParams]):
    def _setup(self, **kwargs) -> None:
        self._generator = None
        self._fixed_kwargs = {}
        for field in self.params.generator_kwargs:
            if getattr(self.params, field) is not None:
                self._fixed_kwargs[field] = getattr(self.params, field)
        if people_gen_resource := kwargs.get("people_gen_resource"):
            if self.params.people_gen_key not in people_gen_resource:
                raise ValueError(f"Person generator with key {self.params.people_gen_key} not found.")
            self.set_generator(people_gen_resource[self.params.people_gen_key])

    def set_generator(self, generator: PeopleGen) -> None:
        self._generator = generator

    def sample(self, num_samples: int) -> NumpyArray1dT:
        if self._generator is None:
            raise ValueError("Generator not set. Please setup generator before sampling.")

        samples = np.array(self._generator.generate(num_samples, **self._fixed_kwargs))
        if len(samples) < num_samples:
            raise ValueError(f"Only {len(samples)} samples could be generated given constraints {self._fixed_kwargs}.")
        return samples


@SamplerRegistry.register(SamplerType.TIMEDELTA)
class TimeDeltaSampler(DatetimeFormatMixin, Sampler[TimeDeltaSamplerParams]):
    def get_required_column_names(self) -> tuple[str, ...]:
        return (self.params.reference_column_name,)

    def inject_data_column(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        index: list[int] | None = None,
    ) -> pd.DataFrame:
        index = slice(None) if index is None else index

        if self.params.reference_column_name not in list(dataframe):
            raise ValueError(f"Columns `{self.params.reference_column_name}` not found in dataset")

        dataframe.loc[index, column_name] = pd.to_datetime(
            dataframe.loc[index, self.params.reference_column_name]
        ) + pd.to_timedelta(self.sample(len(index)), unit=self.params.unit)

        return dataframe

    def sample(self, num_samples: int) -> NumpyArray1dT:
        deltas = self.rng.randint(self.params.dt_min, self.params.dt_max, num_samples)
        return np.array(deltas, dtype=f"timedelta64[{self.params.unit}]")


@SamplerRegistry.register(SamplerType.UUID)
class UUIDSampler(PassthroughMixin, Sampler[UUIDSamplerParams]):
    def sample(self, num_samples: int) -> NumpyArray1dT:
        prefix = self.params.prefix or ""

        uid_list = []
        while len(uid_list) < num_samples:
            uid = (
                f"{prefix}{uuid.uuid4().hex[: self.params.last_index].upper()}"
                if self.params.uppercase
                else f"{prefix}{uuid.uuid4().hex[: self.params.last_index]}"
            )
            if uid not in uid_list:
                uid_list.append(uid)

        return np.array(uid_list)


#########################################
# Scipy Samplers
#########################################


@SamplerRegistry.register(SamplerType.SCIPY)
class ScipySampler(TypeConversionMixin, ScipyStatsSampler[ScipySamplerParams]):
    """Escape hatch sampler to give users access to any scipy.stats distribution."""

    @property
    def distribution(self) -> stats.rv_continuous | stats.rv_discrete:
        return getattr(stats, self.params.dist_name)(**self.params.dist_params)

    def _validate(self) -> None:
        _validate_scipy_distribution(self.params.dist_name, self.params.dist_params)


@SamplerRegistry.register(SamplerType.BERNOULLI)
class BernoulliSampler(TypeConversionMixin, ScipyStatsSampler[BernoulliSamplerParams]):
    @property
    def distribution(self) -> stats.rv_discrete:
        return stats.bernoulli(p=self.params.p)


@SamplerRegistry.register(SamplerType.BERNOULLI_MIXTURE)
class BernoulliMixtureSampler(TypeConversionMixin, Sampler[BernoulliMixtureSamplerParams]):
    def sample(self, num_samples: int) -> NumpyArray1dT:
        return stats.bernoulli(p=self.params.p).rvs(size=num_samples) * getattr(stats, self.params.dist_name)(
            **self.params.dist_params
        ).rvs(size=num_samples)

    def _validate(self) -> None:
        _validate_scipy_distribution(self.params.dist_name, self.params.dist_params)


@SamplerRegistry.register(SamplerType.BINOMIAL)
class BinomialSampler(TypeConversionMixin, ScipyStatsSampler[BinomialSamplerParams]):
    @property
    def distribution(self) -> stats.rv_discrete:
        return stats.binom(n=self.params.n, p=self.params.p)


@SamplerRegistry.register(SamplerType.GAUSSIAN)
class GaussianSampler(TypeConversionMixin, ScipyStatsSampler[GaussianSamplerParams]):
    @property
    def distribution(self) -> stats.rv_continuous:
        return stats.norm(loc=self.params.mean, scale=self.params.stddev)


@SamplerRegistry.register(SamplerType.POISSON)
class PoissonSampler(TypeConversionMixin, ScipyStatsSampler[PoissonSamplerParams]):
    @property
    def distribution(self) -> stats.rv_discrete:
        return stats.poisson(mu=self.params.mean)


@SamplerRegistry.register(SamplerType.UNIFORM)
class UniformSampler(TypeConversionMixin, ScipyStatsSampler[UniformSamplerParams]):
    @property
    def distribution(self) -> stats.rv_continuous:
        return stats.uniform(loc=self.params.low, scale=self.params.high - self.params.low)


###################################################
# Helper functions for loading sources in isolation
###################################################


def load_sampler(sampler_type: SamplerType, **params) -> DataSource:
    """Load a data source from a source type and parameters."""
    return SamplerRegistry.validate_sampler_type(sampler_type)(params=params)


def _validate_scipy_distribution(dist_name: str, dist_params: dict) -> None:
    if not hasattr(stats, dist_name):
        raise InvalidSamplerParamsError(f"Distribution {dist_name} not found in scipy.stats")
    if not hasattr(getattr(stats, dist_name), "rvs"):
        raise InvalidSamplerParamsError(
            f"Distribution {dist_name} does not have a `rvs` method, which is required for sampling."
        )
    try:
        getattr(stats, dist_name)(**dist_params)
    except Exception:
        raise InvalidSamplerParamsError(
            f"Distribution parameters {dist_params} are not a valid for distribution '{dist_name}'"
        )
