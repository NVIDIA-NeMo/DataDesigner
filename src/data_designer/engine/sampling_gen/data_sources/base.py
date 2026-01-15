# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from numpy.typing import NDArray

from data_designer.config.sampler_params import SamplerParamsT
from data_designer.engine.sampling_gen.utils import check_random_state
from data_designer.lazy_heavy_imports import np, pd, scipy

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import scipy

NumpyArray1dT = NDArray[Any]
RadomStateT = int | np.random.RandomState

GenericParamsT = TypeVar("GenericParamsT", bound=SamplerParamsT)

###########################################################
# Processing Mixins
# -----------------
# These mixins are used to apply pre and post processing
# to the data source output. At the moment, the only
# processing that is applied is an optional type/format
# conversion of the output data.
#
# Preprocessing: Applied *before* constraints are applied.
# Postprocessing: Applied at the end of dataset generation.
#
# IMPORTANT: These are only applied when the data are
# being injected into a DataFrame by the DatasetGenerator.
###########################################################


class PassthroughMixin:
    @staticmethod
    def preproc(series: pd.Series, convert_to: str) -> pd.Series:
        return series

    @staticmethod
    def postproc(series: pd.Series, convert_to: str) -> pd.Series:
        return series

    @staticmethod
    def validate_data_conversion(convert_to: str | None) -> None:
        pass


class TypeConversionMixin:
    """Converts the data type of the output data.

    This mixin applies the same conversion to both the pre and post
    processing steps. The preprocessing is needed to ensure constraints
    are applied to the correct data type. The postprocessing is needed
    to ensure the final dtype is correct. For example, if the user wants an
    `int`, we need to convert to `int` before applying constraints, but
    the ints will be converted back to floats when injected into the
    dataframe (assuming some rows are non-int values). We therefore need
    to convert back to `int` after all constraints have been applied.
    """

    @staticmethod
    def preproc(series: pd.Series, convert_to: str) -> pd.Series:
        if convert_to is not None:
            if convert_to == "int":
                series = series.round()
            return series.astype(convert_to)
        return series

    @staticmethod
    def postproc(series: pd.Series, convert_to: str | None) -> pd.Series:
        if convert_to is not None:
            if convert_to == "int":
                series = series.round()
            return series.astype(convert_to)
        return series

    @staticmethod
    def validate_data_conversion(convert_to: str | None) -> None:
        if convert_to is not None and convert_to not in ["float", "int", "str"]:
            raise ValueError(f"Invalid `convert_to` value: {convert_to}. Must be one of: [float, int, str]")


class DatetimeFormatMixin:
    @staticmethod
    def preproc(series: pd.Series, convert_to: str | None) -> pd.Series:
        return series

    @staticmethod
    def postproc(series: pd.Series, convert_to: str | None) -> pd.Series:
        if convert_to is not None:
            return series.dt.strftime(convert_to)
        if series.dt.month.nunique() == 1:
            return series.apply(lambda dt: dt.year).astype(str)
        if series.dt.day.nunique() == 1:
            return series.apply(lambda dt: dt.strftime("%Y-%m"))
        if series.dt.hour.sum() > 0 or series.dt.minute.sum() > 0:
            return series.apply(lambda dt: dt.isoformat()).astype(str)
        if series.dt.second.sum() == 0:
            return series.apply(lambda dt: dt.date()).astype(str)
        return series.apply(lambda dt: dt.isoformat()).astype(str)

    @staticmethod
    def validate_data_conversion(convert_to: str | None) -> None:
        if convert_to is not None:
            try:
                pd.to_datetime(pd.to_datetime("2012-12-21").strftime(convert_to))
            except Exception as e:
                raise ValueError(f"Invalid datetime format: {convert_to}. {e}")


###########################################################
# Base Data Source Classes
###########################################################


class DataSource(ABC, Generic[GenericParamsT]):
    def __init__(
        self,
        params: GenericParamsT,
        random_state: RadomStateT | None = None,
        **kwargs,
    ):
        self.rng = check_random_state(random_state)
        self.params = self.get_param_type().model_validate(params)
        self._setup(**kwargs)
        self._validate()

    @classmethod
    def get_param_type(cls) -> type[GenericParamsT]:
        return cls.__orig_bases__[-1].__args__[0]

    @abstractmethod
    def inject_data_column(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        index: list[int] | None = None,
    ) -> pd.DataFrame: ...

    @staticmethod
    @abstractmethod
    def preproc(series: pd.Series) -> pd.Series: ...

    @staticmethod
    @abstractmethod
    def postproc(series: pd.Series, convert_to: str | None) -> pd.Series: ...

    @staticmethod
    @abstractmethod
    def validate_data_conversion(convert_to: str | None) -> None: ...

    def get_required_column_names(self) -> tuple[str, ...]:
        return tuple()

    def _setup(self, **kwargs) -> None:
        pass

    def _validate(self) -> None:
        pass


class Sampler(DataSource[GenericParamsT], ABC):
    def _recast_types_if_needed(
        self,
        index: list[int] | slice,
        column_name: str,
        sample: NumpyArray1dT,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        # Type may be different if the column has mixed types / NaNs.
        if column_name in dataframe.columns:
            dtype = sample.dtype
            if dtype != dataframe.loc[index, column_name].dtype:
                dataframe = dataframe.astype({column_name: dtype}, errors="ignore")
        return dataframe

    def inject_data_column(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        index: list[int] | None = None,
    ) -> pd.DataFrame:
        index = slice(None) if index is None else index

        if len(index) == 0:
            return dataframe

        sample = self.sample(len(index))

        # Try recasting before assigning the sample to the dataframe, since setting an item
        # of incompatible dtype is deprecated and will raise an error in future versions.
        dataframe = self._recast_types_if_needed(index, column_name, sample, dataframe)
        dataframe.loc[index, column_name] = sample

        # Recast again in case the assignment led to inconsistencies (e.g., funny business from NaNs).
        dataframe = self._recast_types_if_needed(index, column_name, sample, dataframe)

        return dataframe

    @abstractmethod
    def sample(self, num_samples: int) -> NumpyArray1dT: ...


class ScipyStatsSampler(Sampler[GenericParamsT], ABC):
    @property
    @abstractmethod
    def distribution(self) -> scipy.stats.rv_continuous | scipy.stats.rv_discrete: ...

    def sample(self, num_samples: int) -> NumpyArray1dT:
        return self.distribution.rvs(size=num_samples, random_state=self.rng)
