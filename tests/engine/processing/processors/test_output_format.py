# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import OutputFormatProcessorConfig
from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.processing.processors.output_format import OutputFormatProcessor


@pytest.fixture
def stub_processor_config() -> OutputFormatProcessorConfig:
    return OutputFormatProcessorConfig(
        build_stage=BuildStage.POST_BATCH,
        template='{"text": "{{ col1 }}", "value": "{{ col2 }}"}',
        name="test_output_format",
    )


@pytest.fixture
def stub_processor(stub_processor_config: OutputFormatProcessorConfig) -> OutputFormatProcessor:
    mock_resource_provider = Mock()
    mock_artifact_storage = Mock()
    mock_artifact_storage.write_batch_to_parquet_file = Mock()
    mock_resource_provider.artifact_storage = mock_artifact_storage

    processor = OutputFormatProcessor(
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
    metadata = OutputFormatProcessor.metadata()

    assert metadata.name == "output_format"
    assert metadata.description == "Format the dataset using a Jinja2 template."
    assert metadata.required_resources is None


def test_process_returns_original_dataframe(
    stub_processor: OutputFormatProcessor, stub_simple_dataframe: pd.DataFrame
) -> None:
    result = stub_processor.process(stub_simple_dataframe, current_batch_number=0)
    pd.testing.assert_frame_equal(result, stub_simple_dataframe)


def test_process_writes_formatted_output_to_parquet(
    stub_processor: OutputFormatProcessor, stub_simple_dataframe: pd.DataFrame
) -> None:
    # Capture the formatted dataframe that is written to parquet
    written_dataframe: pd.DataFrame | None = None

    def capture_dataframe(batch_number: int, dataframe: pd.DataFrame, batch_stage: BatchStage, subfolder: str) -> None:
        nonlocal written_dataframe
        written_dataframe = dataframe

    stub_processor.artifact_storage.write_batch_to_parquet_file.side_effect = capture_dataframe

    # Process the dataframe
    result = stub_processor.process(stub_simple_dataframe, current_batch_number=0)

    # Verify the original dataframe is returned
    pd.testing.assert_frame_equal(result, stub_simple_dataframe)

    # Verify write_batch_to_parquet_file was called with correct parameters
    stub_processor.artifact_storage.write_batch_to_parquet_file.assert_called_once()
    call_args = stub_processor.artifact_storage.write_batch_to_parquet_file.call_args

    assert call_args.kwargs["batch_number"] == 0
    assert call_args.kwargs["batch_stage"] == BatchStage.PROCESSORS_OUTPUTS
    assert call_args.kwargs["subfolder"] == "test_output_format"

    # Verify the formatted dataframe has the correct structure
    assert written_dataframe is not None
    assert list(written_dataframe.columns) == ["formatted_output"]
    assert len(written_dataframe) == 4

    # Verify the formatted content
    expected_formatted_output = [
        '{"text": "hello", "value": "1"}',
        '{"text": "world", "value": "2"}',
        '{"text": "test", "value": "3"}',
        '{"text": "data", "value": "4"}',
    ]

    for i, expected in enumerate(expected_formatted_output):
        actual = written_dataframe.iloc[i]["formatted_output"]
        # Parse both as JSON to compare structure (ignoring whitespace differences)
        assert json.loads(actual) == json.loads(expected), f"Row {i} mismatch: {actual} != {expected}"


def test_process_without_batch_number_does_not_write(
    stub_processor: OutputFormatProcessor, stub_simple_dataframe: pd.DataFrame
) -> None:
    # Process without batch number (preview mode)
    result = stub_processor.process(stub_simple_dataframe, current_batch_number=None)

    # Verify the original dataframe is returned
    pd.testing.assert_frame_equal(result, stub_simple_dataframe)

    # Verify write_batch_to_parquet_file was NOT called
    stub_processor.artifact_storage.write_batch_to_parquet_file.assert_not_called()


def test_process_with_json_serialized_values(stub_processor: OutputFormatProcessor) -> None:
    # Test with JSON-serialized values in dataframe
    df_with_json = pd.DataFrame(
        {
            "col1": ["hello", "world"],
            "col2": ['{"nested": "value1"}', '{"nested": "value2"}'],
        }
    )

    written_dataframe: pd.DataFrame | None = None

    def capture_dataframe(batch_number: int, dataframe: pd.DataFrame, batch_stage: BatchStage, subfolder: str) -> None:
        nonlocal written_dataframe
        written_dataframe = dataframe

    stub_processor.artifact_storage.write_batch_to_parquet_file.side_effect = capture_dataframe

    # Process the dataframe
    stub_processor.process(df_with_json, current_batch_number=0)

    # Verify the formatted dataframe was written
    assert written_dataframe is not None
    assert len(written_dataframe) == 2

    # Verify that nested JSON values are properly deserialized in template rendering
    first_output = json.loads(written_dataframe.iloc[0]["formatted_output"])
    assert first_output["text"] == "hello"
    assert first_output["value"] == "{'nested': 'value1'}"
