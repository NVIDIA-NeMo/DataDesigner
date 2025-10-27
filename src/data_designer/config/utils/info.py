# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..sampler_params import SamplerType
from .type_helpers import get_sampler_params
from .visualization import display_sampler_table


class DataDesignerInfo:
    def __init__(self):
        self._sampler_params = get_sampler_params()

    @property
    def sampler_table(self) -> None:
        display_sampler_table(self._sampler_params)

    @property
    def sampler_types(self) -> list[str]:
        return [s.value for s in SamplerType]

    def display_sampler(self, sampler_type: SamplerType) -> None:
        title = f"{SamplerType(sampler_type).value.replace('_', ' ').title()} Sampler"
        display_sampler_table({sampler_type: self._sampler_params[sampler_type]}, title=title)
