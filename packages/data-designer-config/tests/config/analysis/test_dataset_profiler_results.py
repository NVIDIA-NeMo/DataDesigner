# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

from data_designer.config.analysis.column_statistics import GeneralColumnStatistics, MissingValue
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.utils.constants import EPSILON


def test_dataset_profiler_results_creation(sample_dataset_profiler_results):
    """Test that DatasetProfilerResults can be created with valid data."""
    assert sample_dataset_profiler_results.num_records == 1000
    assert sample_dataset_profiler_results.target_num_records == 1000
    assert len(sample_dataset_profiler_results.column_statistics) == 3
    assert sample_dataset_profiler_results.side_effect_column_names == ["side_effect_1", "side_effect_2"]
    assert sample_dataset_profiler_results.column_profiles is None


def test_percent_complete_property(sample_dataset_profiler_results):
    """Test the percent_complete property calculation."""
    # Test normal case
    assert round(sample_dataset_profiler_results.percent_complete, 1) == 100.0

    # Test with different target
    result = DatasetProfilerResults(
        num_records=500,
        target_num_records=1000,
        column_statistics=[
            GeneralColumnStatistics(
                column_name="test",
                num_records=500,
                num_null=0,
                num_unique=500,
                pyarrow_dtype="string",
                simple_dtype="str",
            )
        ],
    )
    assert round(result.percent_complete, 1) == 50.0

    # Test with zero target (should use epsilon)
    result_zero_target = DatasetProfilerResults(
        num_records=100,
        target_num_records=0,
        column_statistics=[
            GeneralColumnStatistics(
                column_name="test",
                num_records=100,
                num_null=0,
                num_unique=100,
                pyarrow_dtype="string",
                simple_dtype="str",
            )
        ],
    )
    expected_percent = 100 * 100 / (0 + EPSILON)
    assert result_zero_target.percent_complete == expected_percent


def test_get_column_statistics_by_type(sample_dataset_profiler_results):
    """Test getting column statistics filtered by type."""
    # Test getting general columns
    general_cols = sample_dataset_profiler_results.get_column_statistics_by_type("seed-dataset")
    assert len(general_cols) == 1
    assert general_cols[0].column_name == "seed_column"

    # Test getting LLM text columns
    llm_text_cols = sample_dataset_profiler_results.get_column_statistics_by_type("llm-text")
    assert len(llm_text_cols) == 1
    assert llm_text_cols[0].column_name == "llm_text_column"

    # Test getting sampler columns
    sampler_cols = sample_dataset_profiler_results.get_column_statistics_by_type("sampler")
    assert len(sampler_cols) == 1
    assert sampler_cols[0].column_name == "sampler_column"

    # Test getting non-existent type
    non_existent = sample_dataset_profiler_results.get_column_statistics_by_type("non-existent")
    assert len(non_existent) == 0


def test_to_report_method(sample_dataset_profiler_results):
    """Test the to_report method calls generate_analysis_report."""
    with patch("data_designer.config.analysis.dataset_profiler.generate_analysis_report") as mock_generate:
        # Test with save path as None
        sample_dataset_profiler_results.to_report()
        mock_generate.assert_called_once_with(
            sample_dataset_profiler_results,
            None,
            include_sections=None,
        )

        # Test with save path as string
        save_path = "/tmp/test_report.html"
        sample_dataset_profiler_results.to_report(save_path)
        mock_generate.assert_called_with(
            sample_dataset_profiler_results,
            save_path,
            include_sections=None,
        )

        # Test with save path as Path object
        save_path = Path("/tmp/test_report.html")
        sample_dataset_profiler_results.to_report(save_path)
        mock_generate.assert_called_with(
            sample_dataset_profiler_results,
            save_path,
            include_sections=None,
        )


def test_dataset_profiler_results_with_missing_values():
    """Test DatasetProfilerResults with columns containing missing values."""
    column_with_missing = GeneralColumnStatistics(
        column_name="missing_column",
        num_records=MissingValue.CALCULATION_FAILED,
        num_null=MissingValue.CALCULATION_FAILED,
        num_unique=MissingValue.CALCULATION_FAILED,
        pyarrow_dtype="string",
        simple_dtype="str",
    )

    result = DatasetProfilerResults(
        num_records=100,
        target_num_records=100,
        column_statistics=[column_with_missing],
    )

    assert result.num_records == 100
    assert len(result.column_statistics) == 1
    assert result.column_statistics[0].column_name == "missing_column"


def test_dataset_profiler_results_from_dict():
    """Test creating DatasetProfilerResults from dictionary."""
    data = {
        "num_records": 200,
        "target_num_records": 200,
        "column_statistics": [
            {
                "column_name": "test_col",
                "num_records": 200,
                "num_null": 10,
                "num_unique": 190,
                "pyarrow_dtype": "string",
                "simple_dtype": "str",
                "column_type": "general",
            }
        ],
        "side_effect_column_names": None,
        "column_profiles": None,
    }

    result = DatasetProfilerResults.model_validate(data)
    assert result.num_records == 200
    assert result.target_num_records == 200
    assert len(result.column_statistics) == 1
    assert result.column_statistics[0].column_name == "test_col"
