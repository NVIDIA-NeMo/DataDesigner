# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import ToJsonlProcessorConfig
from data_designer.engine.processing.processors.to_jsonl import ToJsonlProcessor


@pytest.fixture
def stub_processor_config() -> ToJsonlProcessorConfig:
    return ToJsonlProcessorConfig(
        build_stage=BuildStage.POST_BATCH,
        template={"text": "{{ col1 }}", "value": "{{ col2 }}"},
        folder_name="jsonl_output",
        fraction_per_file={"train.jsonl": 0.75, "validation.jsonl": 0.25},
    )


@pytest.fixture
def stub_processor(stub_processor_config: ToJsonlProcessorConfig, tmp_path: Path) -> ToJsonlProcessor:
    mock_resource_provider = Mock()
    mock_artifact_storage = Mock()
    mock_artifact_storage.move_to_outputs = Mock()
    mock_resource_provider.artifact_storage = mock_artifact_storage

    processor = ToJsonlProcessor(
        config=stub_processor_config,
        resource_provider=mock_resource_provider,
    )
    return processor


@pytest.fixture
def stub_simple_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col1": ["hello", "world", "test", "data"],
            "col2": [1, 2, 3, 4],
        }
    )


def test_metadata() -> None:
    metadata = ToJsonlProcessor.metadata()

    assert metadata.name == "to_jsonl"
    assert metadata.description == "Save formatted dataset as JSONL files."
    assert metadata.required_resources is None


def test_template_as_string(stub_processor: ToJsonlProcessor) -> None:
    template_str = stub_processor.template_as_string
    assert isinstance(template_str, str)
    assert json.loads(template_str) == {"text": "{{ col1 }}", "value": "{{ col2 }}"}


def test_get_stop_index_per_file(stub_processor: ToJsonlProcessor) -> None:
    stub_processor.config.fraction_per_file = {"train.jsonl": 0.8, "val.jsonl": 0.15, "test.jsonl": 0.05}
    result = stub_processor._get_stop_index_per_file(100)

    assert result == {"train.jsonl": 80, "val.jsonl": 95, "test.jsonl": 100}


def test_process_returns_original_dataframe(
    stub_processor: ToJsonlProcessor, stub_simple_dataframe: pd.DataFrame
) -> None:
    result = stub_processor.process(stub_simple_dataframe)
    pd.testing.assert_frame_equal(result, stub_simple_dataframe)


def test_process_writes_correct_content_to_files(
    stub_processor: ToJsonlProcessor, stub_simple_dataframe: pd.DataFrame
) -> None:
    stub_processor.config.fraction_per_file = {"train.jsonl": 0.75, "validation.jsonl": 0.25}

    # Capture the content of the files that are written to the outputs folder
    file_contents: dict[str, str] = {}

    def capture_file_content(file_path: Path, folder_name: str) -> None:
        with open(file_path, "r") as f:
            file_contents[file_path.name] = f.read()

    stub_processor.artifact_storage.move_to_outputs.side_effect = capture_file_content

    # Process the dataframe and write the files to the outputs folder
    with patch("data_designer.engine.processing.processors.to_jsonl.logger"):
        stub_processor.process(stub_simple_dataframe)

    # Check that the files were moved with the correct names
    assert stub_processor.artifact_storage.move_to_outputs.call_count == 2

    assert "train.jsonl" in file_contents
    assert "validation.jsonl" in file_contents

    # Check that the files contain the correct content
    train_lines = file_contents["train.jsonl"].strip().split("\n") if file_contents["train.jsonl"].strip() else []
    validation_lines = (
        file_contents["validation.jsonl"].strip().split("\n") if file_contents["validation.jsonl"].strip() else []
    )

    assert len(train_lines) == 3, f"Expected 3 lines in train.jsonl, got {len(train_lines)}"
    assert len(validation_lines) == 1, f"Expected 1 line in validation.jsonl, got {len(validation_lines)}"

    expected_train_data = [
        {"text": "hello", "value": "1"},
        {"text": "world", "value": "2"},
        {"text": "test", "value": "3"},
    ]

    for i, line in enumerate(train_lines):
        parsed = json.loads(line)
        assert parsed == expected_train_data[i], f"Train line {i} mismatch: {parsed} != {expected_train_data[i]}"

    expected_validation_data = [{"text": "data", "value": "4"}]

    for i, line in enumerate(validation_lines):
        parsed = json.loads(line)
        assert parsed == expected_validation_data[i], (
            f"Validation line {i} mismatch: {parsed} != {expected_validation_data[i]}"
        )
