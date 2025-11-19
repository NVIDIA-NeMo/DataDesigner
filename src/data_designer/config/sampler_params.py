# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal, Optional, Union

import pandas as pd
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self, TypeAlias

from .base import ConfigBase
from .utils.constants import (
    AVAILABLE_LOCALES,
    DEFAULT_AGE_RANGE,
    LOCALES_WITH_MANAGED_DATASETS,
    MAX_AGE,
    MIN_AGE,
)


class SamplerType(str, Enum):
    BERNOULLI = "bernoulli"
    BERNOULLI_MIXTURE = "bernoulli_mixture"
    BINOMIAL = "binomial"
    CATEGORY = "category"
    DATETIME = "datetime"
    GAUSSIAN = "gaussian"
    PERSON = "person"
    PERSON_FROM_FAKER = "person_from_faker"
    POISSON = "poisson"
    SCIPY = "scipy"
    SUBCATEGORY = "subcategory"
    TIMEDELTA = "timedelta"
    UNIFORM = "uniform"
    UUID = "uuid"


#########################################
# Sampler Parameters
#########################################


class CategorySamplerParams(ConfigBase):
    values: list[Union[str, int, float]] = Field(
        ...,
        min_length=1,
        description="List of possible categorical values that can be sampled from.",
    )
    weights: Optional[list[float]] = Field(
        default=None,
        description=(
            "List of unnormalized probability weights to assigned to each value, in order. "
            "Larger values will be sampled with higher probability."
        ),
    )

    @model_validator(mode="after")
    def _normalize_weights_if_needed(self) -> Self:
        if self.weights is not None:
            self.weights = [w / sum(self.weights) for w in self.weights]
        return self

    @model_validator(mode="after")
    def _validate_equal_lengths(self) -> Self:
        if self.weights and len(self.values) != len(self.weights):
            raise ValueError("'categories' and 'weights' must have the same length")
        return self


class DatetimeSamplerParams(ConfigBase):
    start: str = Field(..., description="Earliest possible datetime for sampling range, inclusive.")
    end: str = Field(..., description="Latest possible datetime for sampling range, inclusive.")
    unit: Literal["Y", "M", "D", "h", "m", "s"] = Field(
        default="D",
        description="Sampling units, e.g. the smallest possible time interval between samples.",
    )

    @field_validator("start", "end")
    @classmethod
    def _validate_param_is_datetime(cls, value: str) -> str:
        try:
            pd.to_datetime(value)
        except ValueError:
            raise ValueError(f"Invalid datetime format: {value}")
        return value


class SubcategorySamplerParams(ConfigBase):
    category: str = Field(..., description="Name of parent category to this subcategory.")
    values: dict[str, list[Union[str, int, float]]] = Field(
        ...,
        description="Mapping from each value of parent category to a list of subcategory values.",
    )


class TimeDeltaSamplerParams(ConfigBase):
    dt_min: int = Field(
        ...,
        ge=0,
        description=("Minimum possible time-delta for sampling range, inclusive. Must be less than `dt_max`."),
    )
    dt_max: int = Field(
        ...,
        gt=0,
        description=("Maximum possible time-delta for sampling range, exclusive. Must be greater than `dt_min`."),
    )

    reference_column_name: str = Field(
        ...,
        description="Name of an existing datetime column to condition time-delta sampling on.",
    )

    # NOTE: pandas does not support years or months as timedelta units
    # since they are ambiguous. We will need to update the implementation
    # if we need to support these units.
    # see: https://pandas.pydata.org/docs/user_guide/timedeltas.html.
    unit: Literal["D", "h", "m", "s"] = Field(
        default="D",
        description="Sampling units, e.g. the smallest possible time interval between samples.",
    )

    @model_validator(mode="after")
    def _validate_min_less_than_max(self) -> Self:
        if self.dt_min >= self.dt_max:
            raise ValueError("'dt_min' must be less than 'dt_max'")
        return self


class UUIDSamplerParams(ConfigBase):
    prefix: Optional[str] = Field(default=None, description="String prepended to the front of the UUID.")
    short_form: bool = Field(
        default=False,
        description="If true, all UUIDs sampled will be truncated at 8 characters.",
    )
    uppercase: bool = Field(
        default=False,
        description="If true, all letters in the UUID will be capitalized.",
    )

    @property
    def last_index(self) -> int:
        return 8 if self.short_form else 32


#########################################
# Scipy Sampler Parameters
#########################################


class ScipySamplerParams(ConfigBase):
    dist_name: str = Field(..., description="Name of a scipy.stats distribution.")
    dist_params: dict = Field(
        ...,
        description="Parameters of the scipy.stats distribution given in `dist_name`.",
    )
    decimal_places: Optional[int] = Field(
        default=None, description="Number of decimal places to round the sampled values to."
    )


class BinomialSamplerParams(ConfigBase):
    n: int = Field(..., description="Number of trials.")
    p: float = Field(..., description="Probability of success on each trial.", ge=0.0, le=1.0)


class BernoulliSamplerParams(ConfigBase):
    p: float = Field(..., description="Probability of success.", ge=0.0, le=1.0)


class BernoulliMixtureSamplerParams(ConfigBase):
    p: float = Field(
        ...,
        description="Bernoulli distribution probability of success.",
        ge=0.0,
        le=1.0,
    )
    dist_name: str = Field(
        ...,
        description=(
            "Mixture distribution name. Samples will be equal to the "
            "distribution sample with probability `p`, otherwise equal to 0. "
            "Must be a valid scipy.stats distribution name."
        ),
    )
    dist_params: dict = Field(
        ...,
        description="Parameters of the scipy.stats distribution given in `dist_name`.",
    )


class GaussianSamplerParams(ConfigBase):
    mean: float = Field(..., description="Mean of the Gaussian distribution")
    stddev: float = Field(..., description="Standard deviation of the Gaussian distribution")
    decimal_places: Optional[int] = Field(
        default=None, description="Number of decimal places to round the sampled values to."
    )


class PoissonSamplerParams(ConfigBase):
    mean: float = Field(..., description="Mean number of events in a fixed interval.")


class UniformSamplerParams(ConfigBase):
    low: float = Field(..., description="Lower bound of the uniform distribution, inclusive.")
    high: float = Field(..., description="Upper bound of the uniform distribution, inclusive.")
    decimal_places: Optional[int] = Field(
        default=None, description="Number of decimal places to round the sampled values to."
    )


#########################################
# Person Sampler Parameters
#########################################

SexT: TypeAlias = Literal["Male", "Female"]


class PersonSamplerParams(ConfigBase):
    locale: str = Field(
        default="en_US",
        description=(
            "Locale that determines the language and geographic location "
            "that a synthetic person will be sampled from. Must be a locale supported by "
            "a managed Nemotron Personas dataset. Managed datasets exist for the following locales: "
            f"{', '.join(LOCALES_WITH_MANAGED_DATASETS)}."
        ),
    )
    sex: Optional[SexT] = Field(
        default=None,
        description="If specified, then only synthetic people of the specified sex will be sampled.",
    )
    city: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="If specified, then only synthetic people from these cities will be sampled.",
    )
    age_range: list[int] = Field(
        default=DEFAULT_AGE_RANGE,
        description="If specified, then only synthetic people within this age range will be sampled.",
        min_length=2,
        max_length=2,
    )
    select_field_values: Optional[dict[str, list[str]]] = Field(
        default=None,
        description=(
            "Sample synthetic people with the specified field values. This is meant to be a flexible argument for "
            "selecting a subset of the population from the managed dataset. Note that this sampler does not support "
            "rare combinations of field values and will likely fail if your desired subset is not well-represented "
            "in the managed Nemotron Personas dataset. We generally recommend using the `sex`, `city`, and `age_range` "
            "arguments to filter the population when possible."
        ),
        examples=[
            {"state": ["NY", "CA", "OH", "TX", "NV"], "education_level": ["high_school", "some_college", "bachelors"]}
        ],
    )

    with_synthetic_personas: bool = Field(
        default=False,
        description="If True, then append synthetic persona columns to each generated person.",
    )

    @property
    def generator_kwargs(self) -> list[str]:
        """Keyword arguments to pass to the person generator."""
        return [f for f in list(PersonSamplerParams.model_fields) if f != "locale"]

    @property
    def people_gen_key(self) -> str:
        return f"{self.locale}_with_personas" if self.with_synthetic_personas else self.locale

    @field_validator("age_range")
    @classmethod
    def _validate_age_range(cls, value: list[int]) -> list[int]:
        msg_prefix = "'age_range' must be a list of two integers, representing the min and max age."
        if value[0] < MIN_AGE:
            raise ValueError(
                f"{msg_prefix} The first integer (min age) must be greater than or equal to {MIN_AGE}, "
                f"but the first integer provided was {value[0]}."
            )
        if value[1] > MAX_AGE:
            raise ValueError(
                f"{msg_prefix} The second integer (max age) must be less than or equal to {MAX_AGE}, "
                f"but the second integer provided was {value[1]}."
            )
        if value[0] >= value[1]:
            raise ValueError(
                f"{msg_prefix} The first integer (min age) must be less than the second integer (max age), "
                f"but the first integer provided was {value[0]} and the second integer provided was {value[1]}."
            )
        return value

    @model_validator(mode="after")
    def _validate_locale_with_managed_datasets(self) -> Self:
        if self.locale not in LOCALES_WITH_MANAGED_DATASETS:
            raise ValueError(
                "Person sampling from managed datasets is only supported for the following "
                f"locales: {', '.join(LOCALES_WITH_MANAGED_DATASETS)}."
            )
        return self


class PersonFromFakerSamplerParams(ConfigBase):
    locale: str = Field(
        default="en_US",
        description=(
            "Locale string, determines the language and geographic locale "
            "that a synthetic person will be sampled from. E.g, en_US, en_GB, fr_FR, ..."
        ),
    )
    sex: Optional[SexT] = Field(
        default=None,
        description="If specified, then only synthetic people of the specified sex will be sampled.",
    )
    city: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="If specified, then only synthetic people from these cities will be sampled.",
    )
    age_range: list[int] = Field(
        default=DEFAULT_AGE_RANGE,
        description="If specified, then only synthetic people within this age range will be sampled.",
        min_length=2,
        max_length=2,
    )

    @property
    def generator_kwargs(self) -> list[str]:
        """Keyword arguments to pass to the person generator."""
        return [f for f in list(PersonFromFakerSamplerParams.model_fields) if f != "locale"]

    @property
    def people_gen_key(self) -> str:
        return f"{self.locale}_faker"

    @field_validator("age_range")
    @classmethod
    def _validate_age_range(cls, value: list[int]) -> list[int]:
        msg_prefix = "'age_range' must be a list of two integers, representing the min and max age."
        if value[0] < MIN_AGE:
            raise ValueError(
                f"{msg_prefix} The first integer (min age) must be greater than or equal to {MIN_AGE}, "
                f"but the first integer provided was {value[0]}."
            )
        if value[1] > MAX_AGE:
            raise ValueError(
                f"{msg_prefix} The second integer (max age) must be less than or equal to {MAX_AGE}, "
                f"but the second integer provided was {value[1]}."
            )
        if value[0] >= value[1]:
            raise ValueError(
                f"{msg_prefix} The first integer (min age) must be less than the second integer (max age), "
                f"but the first integer provided was {value[0]} and the second integer provided was {value[1]}."
            )
        return value

    @field_validator("locale")
    @classmethod
    def _validate_locale(cls, value: str) -> str:
        if value not in AVAILABLE_LOCALES:
            raise ValueError(
                f"Locale {value!r} is not a supported locale. Supported locales: {', '.join(AVAILABLE_LOCALES)}"
            )
        return value


SamplerParamsT: TypeAlias = Union[
    SubcategorySamplerParams,
    CategorySamplerParams,
    DatetimeSamplerParams,
    PersonSamplerParams,
    PersonFromFakerSamplerParams,
    TimeDeltaSamplerParams,
    UUIDSamplerParams,
    BernoulliSamplerParams,
    BernoulliMixtureSamplerParams,
    BinomialSamplerParams,
    GaussianSamplerParams,
    PoissonSamplerParams,
    UniformSamplerParams,
    ScipySamplerParams,
]


def is_numerical_sampler_type(sampler_type: SamplerType) -> bool:
    return SamplerType(sampler_type) in {
        SamplerType.BERNOULLI_MIXTURE,
        SamplerType.BERNOULLI,
        SamplerType.BINOMIAL,
        SamplerType.GAUSSIAN,
        SamplerType.POISSON,
        SamplerType.SCIPY,
        SamplerType.UNIFORM,
    }
