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
    """Parameters for categorical sampling with optional probability weighting.

    Samples values from a discrete set of categories. When weights are provided, values are
    sampled according to their assigned probabilities. Without weights, uniform sampling is used.

    Attributes:
        values: List of possible categorical values to sample from. Can contain strings, integers,
            or floats. Must contain at least one value.
        weights: Optional unnormalized probability weights for each value. If provided, must be
            the same length as `values`. Weights are automatically normalized to sum to 1.0.
            Larger weights result in higher sampling probability for the corresponding value.
    """

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
    """Parameters for uniform datetime sampling within a specified range.

    Samples datetime values uniformly between a start and end date with a specified granularity.
    The sampling unit determines the smallest possible time interval between consecutive samples.

    Attributes:
        start: Earliest possible datetime for the sampling range (inclusive). Must be a valid
            datetime string parseable by pandas.to_datetime().
        end: Latest possible datetime for the sampling range (inclusive). Must be a valid
            datetime string parseable by pandas.to_datetime().
        unit: Time unit for sampling granularity. Options:
            - "Y": Years
            - "M": Months
            - "D": Days (default)
            - "h": Hours
            - "m": Minutes
            - "s": Seconds
    """

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
    """Parameters for subcategory sampling conditioned on a parent category column.

    Samples subcategory values based on the value of a parent category column. Each parent
    category value maps to its own list of possible subcategory values, enabling hierarchical
    or conditional sampling patterns.

    Attributes:
        category: Name of the parent category column that this subcategory depends on.
            The parent column must be generated before this subcategory column.
        values: Mapping from each parent category value to a list of possible subcategory values.
            Each key must correspond to a value that appears in the parent category column.
    """

    category: str = Field(..., description="Name of parent category to this subcategory.")
    values: dict[str, list[Union[str, int, float]]] = Field(
        ...,
        description="Mapping from each value of parent category to a list of subcategory values.",
    )


class TimeDeltaSamplerParams(ConfigBase):
    """Parameters for sampling time deltas relative to a reference datetime column.

    Samples time offsets within a specified range and adds them to values from a reference
    datetime column. This is useful for generating related datetime columns like order dates
    and delivery dates, or event start times and end times.

    Note:
        Years and months are not supported as timedelta units because they have variable lengths.
        See: [pandas timedelta documentation](https://pandas.pydata.org/docs/user_guide/timedeltas.html)

    Attributes:
        dt_min: Minimum time-delta value (inclusive). Must be non-negative and less than `dt_max`.
            Specified in units defined by the `unit` parameter.
        dt_max: Maximum time-delta value (exclusive). Must be positive and greater than `dt_min`.
            Specified in units defined by the `unit` parameter.
        reference_column_name: Name of an existing datetime column to add the time-delta to.
            This column must be generated before the timedelta column.
        unit: Time unit for the delta values. Options:
            - "D": Days (default)
            - "h": Hours
            - "m": Minutes
            - "s": Seconds
    """

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
    """Parameters for generating UUID (Universally Unique Identifier) values.

    Generates UUID4 (random) identifiers with optional formatting options. UUIDs are useful
    for creating unique identifiers for records, entities, or transactions.

    Attributes:
        prefix: Optional string to prepend to each UUID. Useful for creating namespaced or
            typed identifiers (e.g., "user-", "order-", "txn-").
        short_form: If True, truncates UUIDs to 8 characters (first segment only). Default is False
            for full 32-character UUIDs (excluding hyphens).
        uppercase: If True, converts all hexadecimal letters to uppercase. Default is False for
            lowercase UUIDs.
    """

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
    """Parameters for sampling from any scipy.stats continuous or discrete distribution.

    Provides a flexible interface to sample from the wide range of probability distributions
    available in scipy.stats. This enables advanced statistical sampling beyond the built-in
    distribution types (Gaussian, Uniform, etc.).

    See: [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)

    Attributes:
        dist_name: Name of the scipy.stats distribution to sample from (e.g., "beta", "gamma",
            "lognorm", "expon"). Must be a valid distribution name from scipy.stats.
        dist_params: Dictionary of parameters for the specified distribution. Parameter names
            and values must match the scipy.stats distribution specification (e.g., {"a": 2, "b": 5}
            for beta distribution, {"scale": 1.5} for exponential).
        decimal_places: Optional number of decimal places to round sampled values to. If None,
            values are not rounded.
    """

    dist_name: str = Field(..., description="Name of a scipy.stats distribution.")
    dist_params: dict = Field(
        ...,
        description="Parameters of the scipy.stats distribution given in `dist_name`.",
    )
    decimal_places: Optional[int] = Field(
        default=None, description="Number of decimal places to round the sampled values to."
    )


class BinomialSamplerParams(ConfigBase):
    """Parameters for sampling from a Binomial distribution.

    Samples integer values representing the number of successes in a fixed number of independent
    Bernoulli trials, each with the same probability of success. Commonly used to model the number
    of successful outcomes in repeated experiments.

    Attributes:
        n: Number of independent trials. Must be a positive integer.
        p: Probability of success on each trial. Must be between 0.0 and 1.0 (inclusive).
    """

    n: int = Field(..., description="Number of trials.")
    p: float = Field(..., description="Probability of success on each trial.", ge=0.0, le=1.0)


class BernoulliSamplerParams(ConfigBase):
    """Parameters for sampling from a Bernoulli distribution.

    Samples binary values (0 or 1) representing the outcome of a single trial with a fixed
    probability of success. This is the simplest discrete probability distribution, useful for
    modeling binary outcomes like success/failure, yes/no, or true/false.

    Attributes:
        p: Probability of success (sampling 1). Must be between 0.0 and 1.0 (inclusive).
            The probability of failure (sampling 0) is automatically 1 - p.
    """

    p: float = Field(..., description="Probability of success.", ge=0.0, le=1.0)


class BernoulliMixtureSamplerParams(ConfigBase):
    """Parameters for sampling from a Bernoulli mixture distribution.

    Combines a Bernoulli distribution with another continuous distribution, creating a mixture
    where values are either 0 (with probability 1-p) or sampled from the specified distribution
    (with probability p). This is useful for modeling scenarios with many zero values mixed with
    a continuous distribution of non-zero values.

    Common use cases include modeling sparse events, zero-inflated data, or situations where
    an outcome either doesn't occur (0) or follows a specific distribution when it does occur.

    Attributes:
        p: Probability of sampling from the mixture distribution (non-zero outcome).
            Must be between 0.0 and 1.0 (inclusive). With probability 1-p, the sample is 0.
        dist_name: Name of the scipy.stats distribution to sample from when outcome is non-zero.
            Must be a valid scipy.stats distribution name (e.g., "norm", "gamma", "expon").
        dist_params: Parameters for the specified scipy.stats distribution.
    """

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
    """Parameters for sampling from a Gaussian (Normal) distribution.

    Samples continuous values from a normal distribution characterized by its mean and standard
    deviation. The Gaussian distribution is one of the most commonly used probability distributions,
    appearing naturally in many real-world phenomena due to the Central Limit Theorem.

    Attributes:
        mean: Mean (center) of the Gaussian distribution. This is the expected value and the
            location of the distribution's peak.
        stddev: Standard deviation of the Gaussian distribution. Controls the spread or width
            of the distribution. Must be positive.
        decimal_places: Optional number of decimal places to round sampled values to. If None,
            values are not rounded.
    """

    mean: float = Field(..., description="Mean of the Gaussian distribution")
    stddev: float = Field(..., description="Standard deviation of the Gaussian distribution")
    decimal_places: Optional[int] = Field(
        default=None, description="Number of decimal places to round the sampled values to."
    )


class PoissonSamplerParams(ConfigBase):
    """Parameters for sampling from a Poisson distribution.

    Samples non-negative integer values representing the number of events occurring in a fixed
    interval of time or space. The Poisson distribution is commonly used to model count data
    like the number of arrivals, occurrences, or events per time period.

    The distribution is characterized by a single parameter (mean/rate), and both the mean and
    variance equal this parameter value.

    Attributes:
        mean: Mean number of events in the fixed interval (also called rate parameter λ).
            Must be positive. This represents both the expected value and the variance of the
            distribution.
    """

    mean: float = Field(..., description="Mean number of events in a fixed interval.")


class UniformSamplerParams(ConfigBase):
    """Parameters for sampling from a continuous Uniform distribution.

    Samples continuous values uniformly from a specified range, where every value in the range
    has equal probability of being sampled. This is useful when all values within a range are
    equally likely, such as random percentages, proportions, or unbiased measurements.

    Attributes:
        low: Lower bound of the uniform distribution (inclusive). Can be any real number.
        high: Upper bound of the uniform distribution (inclusive). Must be greater than `low`.
        decimal_places: Optional number of decimal places to round sampled values to. If None,
            values are not rounded and may have many decimal places.
    """

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
    """Parameters for sampling synthetic person data with demographic attributes.

    Generates realistic synthetic person data including names, addresses, phone numbers, and other
    demographic information. Data can be sampled from managed datasets (when available) or generated
    using Faker. The sampler supports filtering by locale, sex, age, geographic location, and can
    optionally include synthetic persona descriptions.

    Attributes:
        locale: Locale string determining the language and geographic region for synthetic people.
            Format: language_COUNTRY (e.g., "en_US", "en_GB", "fr_FR", "de_DE", "es_ES", "ja_JP").
            Defaults to "en_US".
        sex: If specified, filters to only sample people of the specified sex. Options: "Male" or
            "Female". If None, samples both sexes.
        city: If specified, filters to only sample people from the specified city or cities. Can be
            a single city name (string) or a list of city names.
        age_range: Two-element list [min_age, max_age] specifying the age range to sample from
            (inclusive). Defaults to a standard age range. Both values must be between minimum and
            maximum allowed ages.
        state: Only supported for "en_US" locale. Filters to sample people from specified US state(s).
            Must be provided as two-letter state abbreviations (e.g., "CA", "NY", "TX"). Can be a
            single state or a list of states.
        with_synthetic_personas: If True, appends additional synthetic persona columns including
            personality traits, interests, and background descriptions. Only supported for certain
            locales with managed datasets.
        sample_dataset_when_available: If True, samples from curated managed datasets when available
            for the specified locale. If False or unavailable, falls back to Faker-generated data.
            Managed datasets typically provide more realistic and diverse synthetic people.
    """

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
