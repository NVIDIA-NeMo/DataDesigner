# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.errors import DataDesignerError


class SamplingGenError(DataDesignerError):
    """Base exception for all errors in the sampling_gen library."""


class RejectionSamplingError(SamplingGenError):
    """Exception for all errors related to rejection sampling."""


class DataConversionError(SamplingGenError):
    """Exception for all errors related to data conversion."""


class DatasetNotAvailableForLocaleError(SamplingGenError):
    """Exception for all errors related to the dataset not being available for a given locale."""


class ManagedDatasetGeneratorError(SamplingGenError):
    """Exception for all errors related to the managed dataset generator."""
