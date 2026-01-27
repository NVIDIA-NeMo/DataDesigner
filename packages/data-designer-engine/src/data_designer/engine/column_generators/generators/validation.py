# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from data_designer.config.column_configs import ValidationColumnConfig
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.code_lang import SQL_DIALECTS, CodeLang
from data_designer.config.validator_params import ValidatorParamsT, ValidatorType
from data_designer.engine.column_generators.generators.base import ColumnGeneratorFullColumn
from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.engine.errors import DataDesignerRuntimeError
from data_designer.engine.validators import (
    BaseValidator,
    LocalCallableValidator,
    PythonValidator,
    RemoteValidator,
    SQLValidator,
    ValidationResult,
)
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def get_validator_from_params(validator_type: ValidatorType, validator_params: ValidatorParamsT) -> BaseValidator:
    if validator_type == ValidatorType.CODE:
        if validator_params.code_lang == CodeLang.PYTHON:
            return PythonValidator(validator_params)
        elif validator_params.code_lang in SQL_DIALECTS:
            return SQLValidator(validator_params)
    elif validator_type == ValidatorType.REMOTE:
        return RemoteValidator(validator_params)
    else:
        return LocalCallableValidator(validator_params)


class ValidationColumnGenerator(ColumnGeneratorFullColumn[ValidationColumnConfig]):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"🔍 Validating column {self.config.name!r} with {len(data)} records")
        logger.info(f"  |-- target columns: {self.config.target_columns}")
        logger.info(f"  |-- validator type: {self.config.validator_type}")
        logger.info(f"  |-- validator params: {self.config.validator_params}")
        logger.info(f"  |-- batch size: {self.config.batch_size}")

        validator = get_validator_from_params(self.config.validator_type, self.config.validator_params)

        # Check if the target columns are present in the dataset
        missing_columns = set(self.config.target_columns) - set(data.columns)
        if missing_columns:
            raise InvalidConfigError(
                f"Target columns {missing_columns} defined in validation column {self.config.name!r} are missing in dataset"
            )

        # Check whether to pass single columns or multiple columns to the validator
        validate_columns_separately = False
        if self.config.validator_type == ValidatorType.CODE and len(self.config.target_columns) > 1:
            # Code validator expects single column input, so we validate each column separately
            validate_columns_separately = True

            columns_to_validate = [[col] for col in self.config.target_columns]
        else:
            columns_to_validate = [self.config.target_columns]

        outputs_as_dicts = None
        for cols in columns_to_validate:
            # Filter the dataset to only include the target columns, and convert to a list of dictionaries
            records = data[cols].to_dict(orient="records")

            batched_records = [
                records[batch_start : batch_start + self.config.batch_size]
                for batch_start in range(0, len(records), self.config.batch_size)
            ]

            # Run validation in parallel or sequentially, depending on the validator type and parameters
            if (
                self.config.validator_type == ValidatorType.REMOTE
                and self.config.validator_params.max_parallel_requests > 1
            ):
                concatenated_outputs = self._validate_in_parallel(validator, batched_records)
            else:
                concatenated_outputs = []
                for batch in batched_records:
                    concatenated_outputs.extend(self._validate_batch(validator, batch))

            if validate_columns_separately:
                if outputs_as_dicts is None:
                    outputs_as_dicts = [{cols[0]: output.model_dump(mode="json")} for output in concatenated_outputs]
                else:
                    for dict_output in outputs_as_dicts:
                        dict_output[cols[0]] = concatenated_outputs[0].model_dump(mode="json")
            else:
                outputs_as_dicts = [output.model_dump(mode="json") for output in concatenated_outputs]

        validation_results = pd.DataFrame({self.config.name: outputs_as_dicts})
        return pd.concat([data, validation_results], axis=1)

    def _validate_in_parallel(self, validator: BaseValidator, batched_records: list[list[dict]]) -> pd.DataFrame:
        """Run validation in parallel."""

        outputs = [None] * len(batched_records)

        def result_callback(result: ValidationResult, context: dict):
            outputs[context["index"]] = result

        def error_callback(error: Exception, context: dict):
            outputs[context["index"]] = ValidationResult.empty(size=len(batched_records[context["index"]]))

        settings = self.resource_provider.run_config
        with ConcurrentThreadExecutor(
            max_workers=self.config.validator_params.max_parallel_requests,
            column_name=self.config.name,
            result_callback=result_callback,
            error_callback=error_callback,
            shutdown_error_rate=settings.shutdown_error_rate,
            shutdown_error_window=settings.shutdown_error_window,
            disable_early_shutdown=settings.disable_early_shutdown,
        ) as executor:
            for i, batch in enumerate(batched_records):
                executor.submit(lambda batch: self._validate_batch(validator, batch), batch, context={"index": i})

        if any(output is None for output in outputs):
            raise DataDesignerRuntimeError("Validation task failed due to an unexpected error in parallel execution")

        # Concatenate the outputs and convert to a DataFrame
        return sum([output.data for output in outputs], [])

    def _validate_batch(self, validator: BaseValidator, batch: list[dict]) -> ValidationResult:
        try:
            return validator.run_validation(batch)
        except Exception as e:
            error_to_display = str(e).replace("\n", "\n  ")  # add spaces to improve readability
            logger.error(f"Batch could not be validated:\n  {error_to_display}")
            raise e
