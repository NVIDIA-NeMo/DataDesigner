# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from data_designer.errors import DataDesignerError


class DataDesignerRuntimeError(DataDesignerError): ...


class UnknownModelAliasError(DataDesignerError): ...


class UnknownProviderError(DataDesignerError): ...


class NoModelProvidersError(DataDesignerError): ...


class SecretResolutionError(DataDesignerError): ...


class RemoteValidationSchemaError(DataDesignerError): ...


class LocalCallableValidationError(DataDesignerError): ...


class ErrorTrap(BaseModel):
    error_count: int = 0
    task_errors: dict[str, int] = Field(default_factory=dict)

    def _track_error(self, error: DataDesignerError) -> None:
        """
        Track a specific error type.
        """
        error_type = type(error).__name__
        if error_type not in self.task_errors:
            self.task_errors[error_type] = 0
        self.task_errors[error_type] += 1

    def handle_error(self, error: Exception) -> None:
        self.error_count += 1

        if not isinstance(error, DataDesignerError):
            error = DataDesignerError(str(error))

        self._track_error(error)
