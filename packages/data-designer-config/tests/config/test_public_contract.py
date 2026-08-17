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


@pytest.mark.parametrize("filename", ["missing.yaml", "missing.json", "missing.YAML", "missing.JsOn", "[broken.JSON"])
def test_from_config_normalizes_missing_file_error(tmp_path: Path, filename: str) -> None:
    with pytest.raises(InvalidFilePathError) as exc_info:
        DataDesignerConfigBuilder.from_config(str(tmp_path / filename))

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("extension", ["json", "yaml", "yml"])
def test_from_config_accepts_inline_config_ending_in_supported_extension(extension: str) -> None:
    builder = DataDesignerConfigBuilder.from_config(
        f"""
model_configs:
  - alias: stub-model
    model: stub-model
    provider: provider-1
columns:
  - name: category
    column_type: sampler
    sampler_type: category
    params:
      values:
        - value.{extension}"""
    )

    assert builder.get_column_config("category").params.values == [f"value.{extension}"]


def test_from_config_normalizes_undecodable_file_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_bytes(b"\xff")

    with pytest.raises(InvalidFileFormatError) as exc_info:
        DataDesignerConfigBuilder.from_config(config_path)

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize("extension", ["json", "yaml", "yml"])
def test_from_config_normalizes_malformed_config_error(extension: str) -> None:
    with pytest.raises(InvalidFileFormatError):
        DataDesignerConfigBuilder.from_config(f"data_designer: [/tmp/value.{extension}")
