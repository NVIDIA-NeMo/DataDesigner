# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import AncillaryDatasetProcessorConfig
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage, BatchStage
from data_designer.engine.processing.processors.ancillary_dataset import AncillaryDatasetProcessor
from data_designer.engine.resources.resource_provider import ResourceProvider


@pytest.fixture
def stub_processor_config() -> AncillaryDatasetProcessorConfig:
    return AncillaryDatasetProcessorConfig(
        build_stage=BuildStage.POST_BATCH,
        template={"text": "{{ col1 }}", "value": "{{ col2 }}"},
        name="test_ancillary_dataset",
    )


@pytest.fixture
def stub_processor(stub_processor_config: AncillaryDatasetProcessorConfig, stub_resource_provider: ResourceProvider) -> AncillaryDatasetProcessor:
    stub_resource_provider.artifact_storage = Mock()
    stub_resource_provider.artifact_storage.write_batch_to_parquet_file = Mock()

    processor = AncillaryDatasetProcessor(
        config=stub_processor_config,
        resource_provider=stub_resource_provider,
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
    metadata = AncillaryDatasetProcessor.metadata()

    assert metadata.name == "ancillary_dataset_processor"
    assert metadata.description == "Generate an ancillary dataset using a Jinja2 template."
    assert metadata.required_resources is None


def test_process_returns_original_dataframe(
    stub_processor: AncillaryDatasetProcessor, stub_sample_dataframe: pd.DataFrame
) -> None:
    result = stub_processor.process(stub_sample_dataframe, current_batch_number=0)
    pd.testing.assert_frame_equal(result, stub_sample_dataframe)


def test_process_writes_formatted_output_to_parquet(
    stub_processor: AncillaryDatasetProcessor, stub_sample_dataframe: pd.DataFrame
) -> None:
    # Process the dataframe
    result = stub_processor.process(stub_sample_dataframe, current_batch_number=0)

    # Verify the original dataframe is returned
    pd.testing.assert_frame_equal(result, stub_sample_dataframe)

    # Verify write_batch_to_parquet_file was called with correct parameters
    stub_processor.artifact_storage.write_batch_to_parquet_file.assert_called_once()
    call_args = stub_processor.artifact_storage.write_batch_to_parquet_file.call_args

    assert call_args.kwargs["batch_number"] == 0
    assert call_args.kwargs["batch_stage"] == BatchStage.PROCESSORS_OUTPUTS
    assert call_args.kwargs["subfolder"] == "test_ancillary_dataset"

    # Verify the formatted dataframe has the correct structure
    written_dataframe: pd.DataFrame = call_args.kwargs["dataframe"]

    assert written_dataframe is not None
    assert len(written_dataframe) == 4
    assert len(written_dataframe.columns) == 2
    assert list(written_dataframe.columns) == ["text", "value"]

    # Verify the formatted content
    expected_formatted_output = [
        f'{{"text": "{stub_sample_dataframe.iloc[i]["col1"]}", "value": "{stub_sample_dataframe.iloc[i]["col2"]}"}}'
        for i in range(len(stub_sample_dataframe))
    ]

    for i, expected in enumerate(expected_formatted_output):
        actual = json.dumps(written_dataframe.iloc[i].to_dict())
        # Parse both as JSON to compare structure (ignoring whitespace differences)
        assert json.loads(actual) == json.loads(expected), f"Row {i} mismatch: {actual} != {expected}"


def test_process_without_batch_number_does_not_write(
    stub_processor: AncillaryDatasetProcessor, stub_sample_dataframe: pd.DataFrame
) -> None:
    # Process without batch number (preview mode)
    result = stub_processor.process(stub_sample_dataframe, current_batch_number=None)

    # Verify the original dataframe is returned
    pd.testing.assert_frame_equal(result, stub_sample_dataframe)

    # Verify write_batch_to_parquet_file was NOT called
    stub_processor.artifact_storage.write_batch_to_parquet_file.assert_not_called()


def test_process_with_json_serialized_values(stub_processor: AncillaryDatasetProcessor) -> None:
    # Test with JSON-serialized values in dataframe
    df_with_json = pd.DataFrame(
        {
            "col1": ["hello", "world"],
            "col2": ['{"nested": "value1"}', '{"nested": "value2"}'],
        }
    )

    # Process the dataframe
    stub_processor.process(df_with_json, current_batch_number=0)
    written_dataframe: pd.DataFrame = stub_processor.artifact_storage.write_batch_to_parquet_file.call_args.kwargs["dataframe"]

    # Verify the formatted dataframe was written
    assert written_dataframe is not None
    assert len(written_dataframe) == 2

    # Verify that nested JSON values are properly deserialized in template rendering
    first_output = written_dataframe.iloc[0].to_dict()
    assert first_output["text"] == "hello"
    assert first_output["value"] == "{'nested': 'value1'}"
