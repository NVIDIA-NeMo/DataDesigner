# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import field_serializer, model_validator
from typing_extensions import Self

from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.sampler_params import SamplerParamsT, SamplerType
from data_designer.engine.sampling_gen.data_sources.base import DataSource
from data_designer.engine.sampling_gen.data_sources.sources import SamplerRegistry
from data_designer.engine.sampling_gen.jinja_utils import extract_column_names_from_expression


class ConditionalDataColumn(SamplerColumnConfig):
    @property
    def _negative_condition(self) -> str:
        conditions = list(self.conditional_params.keys())
        return "not (" + " or ".join([f"({c})" for c in conditions]) + ")"

    @property
    def conditions(self) -> list[str]:
        c = list(self.conditional_params.keys())
        return c + [self._negative_condition] if len(c) > 0 else ["..."]

    @property
    def conditional_column_names(self) -> set[str]:
        names = set()
        for condition in self.conditional_params.keys():
            names.update(extract_column_names_from_expression(condition))
        return names

    @field_serializer("sampler_type")
    def serialize_sampler_type(self, sampler_type: SamplerType) -> str:
        return SamplerType(sampler_type).value

    @field_serializer("params")
    def serialize_params(self, params: SamplerParamsT) -> dict:
        return params.model_dump()

    @field_serializer("conditional_params")
    def serialize_conditional_params(self, conditional_params: dict[str, SamplerParamsT]) -> dict:
        for condition, params in conditional_params.items():
            conditional_params[condition] = params.model_dump()
        return conditional_params

    @model_validator(mode="before")
    @classmethod
    def validate_params_with_type(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "sampler_type" not in data:
            return data
        if isinstance(data["sampler_type"], str):
            if not SamplerRegistry.is_registered(data["sampler_type"]):
                raise ValueError(
                    f"Invalid sampler type: {data['sampler_type']}. Available samplers: {[s.value for s in SamplerType]}"
                )
        if "params" in data:
            data["params"] = SamplerRegistry.get_sampler(data["sampler_type"])(params=data["params"]).params
        if "conditional_params" in data:
            for condition, params in data["conditional_params"].items():
                data["conditional_params"][condition] = SamplerRegistry.get_sampler(data["sampler_type"])(
                    params=params
                ).params
        return data

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        self.params = SamplerRegistry.validate_sampler_type(self.sampler_type)(params=self.params).params
        return self

    @model_validator(mode="after")
    def validate_data_conversion(self) -> Self:
        self.get_default_sampler().validate_data_conversion(self.convert_to)
        return self

    @model_validator(mode="after")
    def validate_conditional_params(self) -> Self:
        for condition, params in self.conditional_params.items():
            self.conditional_params[condition] = SamplerRegistry.get_sampler(self.sampler_type)(params=params).params
        return self

    def get_default_sampler(self, **kwargs) -> DataSource:
        return self.get_sampler("...", **kwargs)

    def get_sampler(self, condition: str, **kwargs) -> DataSource:
        if condition in ["...", self._negative_condition]:
            return SamplerRegistry.get_sampler(self.sampler_type)(self.params, **kwargs)
        return SamplerRegistry.get_sampler(self.sampler_type)(self.conditional_params[condition], **kwargs)
