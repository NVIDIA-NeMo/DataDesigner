# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.sampling_gen.errors import SamplingGenError


class InvalidSamplerParamsError(SamplingGenError): ...


class PersonSamplerConstraintsError(SamplingGenError): ...
