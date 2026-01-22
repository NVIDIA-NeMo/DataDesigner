# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from data_designer.config.validator_params import LocalCallableValidatorParams
from data_designer.engine.errors import LocalCallableValidationError
from data_designer.engine.processing.gsonschema.validators import validate
from data_designer.engine.validators.base import BaseValidator, ValidationOutput, ValidationResult
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class LocalCallableValidator(BaseValidator):
    def __init__(self, config: LocalCallableValidatorParams):
        self.validation_function = config.validation_function
        self.output_schema = config.output_schema

    def run_validation(self, data: list[dict]) -> ValidationResult:
        df = pd.DataFrame(data)

        try:
            result_as_df = self.validation_function(df)
        except Exception as e:
            logger.error(f"Callback validator failed: {e}")
            raise LocalCallableValidationError(str(e))

        records = result_as_df.to_dict(orient="records")
        result = ValidationResult(data=[ValidationOutput.model_validate(record) for record in records])
        if self.output_schema:
            validate(result.model_dump(mode="json"), self.output_schema, no_extra_properties=True)
        return result
