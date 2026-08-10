# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.config.errors as config_errors
from data_designer.config import (
    DataDesignerConfigBuilder,
    InvalidConfigError,
    InvalidFileFormatError,
    InvalidFilePathError,
)


def test_invalid_config_error_is_publicly_exported() -> None:
    assert InvalidConfigError is config_errors.InvalidConfigError


def test_builder_file_errors_are_publicly_exported() -> None:
    assert InvalidFileFormatError is config_errors.InvalidFileFormatError
    assert InvalidFilePathError is config_errors.InvalidFilePathError


@pytest.mark.parametrize("filename", ["missing.yaml", "missing.json", "missing.YAML", "missing.JsOn"])
def test_from_config_normalizes_missing_file_error(tmp_path: Path, filename: str) -> None:
    with pytest.raises(InvalidFilePathError) as exc_info:
        DataDesignerConfigBuilder.from_config(str(tmp_path / filename))

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_from_config_normalizes_malformed_config_error() -> None:
    with pytest.raises(InvalidFileFormatError):
        DataDesignerConfigBuilder.from_config("data_designer: [")
