# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.engine.errors import (
    DataDesignerRuntimeError,
    ErrorTrap,
    RemoteValidationSchemaError,
    SecretResolutionError,
    UnknownModelAliasError,
    UnknownProviderError,
)


def test_error_message():
    test_cases = [
        (DataDesignerRuntimeError, "Runtime error occurred"),
        (UnknownModelAliasError, "Unknown model alias"),
        (UnknownProviderError, "Unknown provider"),
        (SecretResolutionError, "Secret resolution failed"),
        (RemoteValidationSchemaError, "Remote validation schema error"),
    ]

    for error_class, message in test_cases:
        error = error_class(message)
        assert str(error) == message

    error = DataDesignerRuntimeError()
    assert str(error) == ""


def test_error_trap_custom_values():
    task_errors = {"TestError": 5, "AnotherError": 3}
    error_trap = ErrorTrap(error_count=8, task_errors=task_errors)

    assert error_trap.error_count == 8
    assert error_trap.task_errors == task_errors


def test_error_trap_track_error():
    error_trap = ErrorTrap()

    error1 = DataDesignerRuntimeError("Error 1")
    error2 = DataDesignerRuntimeError("Error 2")
    error3 = UnknownModelAliasError("Error 3")

    error_trap.handle_error(error1)
    error_trap.handle_error(error2)
    error_trap.handle_error(error3)

    assert error_trap.error_count == 3
    assert error_trap.task_errors["DataDesignerRuntimeError"] == 2
    assert error_trap.task_errors["UnknownModelAliasError"] == 1


def test_error_trap_model_dump():
    error_trap = ErrorTrap(error_count=5, task_errors={"TestError": 3})

    data = error_trap.model_dump()

    assert data["error_count"] == 5
    assert data["task_errors"] == {"TestError": 3}
