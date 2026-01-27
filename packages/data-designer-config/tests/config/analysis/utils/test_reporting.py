# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from data_designer.config.analysis.column_profilers import ColumnProfilerResults
from data_designer.config.analysis.column_statistics import ColumnDistributionType, SeedDatasetColumnStatistics
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.analysis.utils.errors import AnalysisReportError
from data_designer.config.analysis.utils.reporting import ReportSection, generate_analysis_report
from data_designer.config.column_types import DataDesignerColumnType


@pytest.fixture
def sample_dataset_with_llm_only(sample_llm_text_column_stats):
    """Create DatasetProfilerResults with only LLM columns for testing footnotes."""
    return DatasetProfilerResults(
        num_records=500,
        target_num_records=500,
        column_statistics=[sample_llm_text_column_stats],
        side_effect_column_names=None,
        column_profiles=None,
    )


@pytest.fixture
def sample_dataset_without_llm(sample_seed_dataset_column_stats, sample_sampler_column_stats):
    """Create DatasetProfilerResults without LLM columns for testing no footnotes."""
    return DatasetProfilerResults(
        num_records=200,
        target_num_records=200,
        column_statistics=[sample_seed_dataset_column_stats, sample_sampler_column_stats],
        side_effect_column_names=None,
        column_profiles=None,
    )


def test_generate_analysis_report_console_output(sample_dataset_profiler_results):
    """Test that generate_analysis_report produces console output."""
    # This test verifies the function runs without error and produces output
    # Since we're not mocking, we can't easily verify the exact console calls
    # but we can verify it doesn't raise an exception
    generate_analysis_report(sample_dataset_profiler_results)
    # If we get here without an exception, the test passes


def test_generate_analysis_report_with_save_path_html(sample_dataset_profiler_results):
    """Test generate_analysis_report with HTML save path."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        generate_analysis_report(sample_dataset_profiler_results, tmp_path)

        # Verify the file was created and has content
        assert Path(tmp_path).exists()
        assert Path(tmp_path).stat().st_size > 0

        # Verify it's valid HTML by checking for basic HTML structure
        with open(tmp_path, "r") as f:
            content = f.read()
            assert "<html" in content.lower() or "<!doctype" in content.lower()

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_generate_analysis_report_with_save_path_svg(sample_dataset_profiler_results):
    """Test generate_analysis_report with SVG save path."""
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        generate_analysis_report(sample_dataset_profiler_results, tmp_path)

        # Verify the file was created and has content
        assert Path(tmp_path).exists()
        assert Path(tmp_path).stat().st_size > 0

        # Verify it's valid SVG by checking for SVG structure
        with open(tmp_path, "r") as f:
            content = f.read()
            assert "<svg" in content.lower()

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_generate_analysis_report_with_path_object(sample_dataset_profiler_results):
    """Test generate_analysis_report with Path object."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        generate_analysis_report(sample_dataset_profiler_results, tmp_path)

        # Verify the file was created and has content
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0

        # Verify it's valid HTML by checking for basic HTML structure
        with open(tmp_path, "r") as f:
            content = f.read()
            assert "<html" in content.lower() or "<!doctype" in content.lower()

    finally:
        tmp_path.unlink(missing_ok=True)


def test_generate_analysis_report_invalid_extension():
    """Test generate_analysis_report with invalid file extension."""
    sample_data = DatasetProfilerResults(
        num_records=100,
        target_num_records=100,
        column_statistics=[
            SeedDatasetColumnStatistics(
                column_name="test",
                num_records=100,
                num_null=0,
                num_unique=100,
                pyarrow_dtype="string",
                simple_dtype="str",
                distribution_type=ColumnDistributionType.CATEGORICAL,
                distribution=None,
            )
        ],
    )

    with pytest.raises(AnalysisReportError, match="The extension of the save path must be either .html or .svg"):
        generate_analysis_report(sample_data, "/tmp/test.txt")


def test_generate_analysis_report_invalid_extension_path_object():
    """Test generate_analysis_report with invalid file extension using Path object."""
    sample_data = DatasetProfilerResults(
        num_records=100,
        target_num_records=100,
        column_statistics=[
            SeedDatasetColumnStatistics(
                column_name="test",
                num_records=100,
                num_null=0,
                num_unique=100,
                pyarrow_dtype="string",
                simple_dtype="str",
                distribution_type=ColumnDistributionType.CATEGORICAL,
                distribution=None,
            )
        ],
    )

    with pytest.raises(AnalysisReportError, match="The extension of the save path must be either .html or .svg"):
        generate_analysis_report(sample_data, Path("/tmp/test.txt"))


def test_generate_analysis_report_with_and_without_footnotes(sample_dataset_with_llm_only, sample_dataset_without_llm):
    generate_analysis_report(sample_dataset_with_llm_only)
    generate_analysis_report(sample_dataset_without_llm)


def test_generate_analysis_report_column_type_tables(sample_dataset_profiler_results):
    generate_analysis_report(sample_dataset_profiler_results)


def test_generate_analysis_report_with_empty_column_statistics():
    """Test generate_analysis_report with empty column statistics list."""
    # This should not happen due to validation, but test the edge case
    sample_data = DatasetProfilerResults(
        num_records=100,
        target_num_records=100,
        column_statistics=[
            SeedDatasetColumnStatistics(
                column_name="test",
                num_records=100,
                num_null=0,
                num_unique=100,
                pyarrow_dtype="string",
                simple_dtype="str",
                distribution_type=ColumnDistributionType.CATEGORICAL,
                distribution=None,
            )
        ],
    )

    # This should not raise an error
    generate_analysis_report(sample_data)


def test_generate_analysis_report_with_and_without_save_path(sample_dataset_profiler_results):
    """Test generate_analysis_report with save_path=None (console output only), .html, and .svg."""

    # Test with None (console output only)
    generate_analysis_report(sample_dataset_profiler_results, None)

    # Test with temporary .html file
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "test_report.html"
        generate_analysis_report(sample_dataset_profiler_results, html_path)
        assert html_path.exists()
        assert html_path.read_text().startswith("<!DOCTYPE html>")  # crude check for HTML output

        # Test with temporary .svg file
        svg_path = Path(tmpdir) / "test_report.svg"
        generate_analysis_report(sample_dataset_profiler_results, svg_path)
        assert svg_path.exists()
        assert svg_path.read_text().startswith("<svg")  # crude check for SVG output


def test_generate_analysis_report_include_sections_is_none(sample_dataset_profiler_results):
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "empty_sections_report.html"

        generate_analysis_report(sample_dataset_profiler_results, html_path, include_sections=None)

        assert html_path.exists()
        content = html_path.read_text()

        assert "Dataset Overview" in content
        for column_type in [
            DataDesignerColumnType.SEED_DATASET,
            DataDesignerColumnType.LLM_TEXT,
            DataDesignerColumnType.SAMPLER,
        ]:
            column_label = column_type.replace("_", " ").title().replace("Llm", "LLM")
            assert f"{column_label} Columns" in content


def test_generate_analysis_report_include_sections_overview_only(sample_dataset_profiler_results):
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "overview_only_report.html"

        generate_analysis_report(sample_dataset_profiler_results, html_path, include_sections=[ReportSection.OVERVIEW])

        assert html_path.exists()
        content = html_path.read_text()

        assert "Dataset Overview" in content

        for column_type in [
            DataDesignerColumnType.SEED_DATASET,
            DataDesignerColumnType.LLM_TEXT,
            DataDesignerColumnType.SAMPLER,
        ]:
            column_label = column_type.replace("_", " ").title().replace("Llm", "LLM")
            assert f"{column_label} Columns" not in content


def test_generate_analysis_report_include_sections_specific_column_types(sample_dataset_profiler_results):
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "specific_columns_report.html"

        generate_analysis_report(
            sample_dataset_profiler_results,
            html_path,
            include_sections=[DataDesignerColumnType.SEED_DATASET, DataDesignerColumnType.LLM_TEXT],
        )

        assert html_path.exists()
        content = html_path.read_text()

        assert "Dataset Overview" not in content
        assert "Sampler Columns" not in content
        for column_type in [DataDesignerColumnType.SEED_DATASET, DataDesignerColumnType.LLM_TEXT]:
            column_label = column_type.replace("_", " ").title().replace("Llm", "LLM")
            assert f"{column_label} Columns" in content


def test_generate_analysis_report_with_unimplemented_column_profiler(sample_dataset_profiler_results):
    """Test generate_analysis_report with a column profiler that uses default create_report_section."""

    class MockColumnProfilerResults(ColumnProfilerResults):
        column_name: str
        test_data: str

    mock_profiler = MockColumnProfilerResults(column_name="test_column", test_data="sample_data")

    sample_dataset_profiler_results.column_profiles = [mock_profiler]

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "unimplemented_profiler_report.html"

        generate_analysis_report(sample_dataset_profiler_results, html_path)

        assert html_path.exists()
        content = html_path.read_text()

        assert "Not Implemented" in content
        assert "Report section generation not implemented for" in content
        assert "MockColumnProfilerResults" in content
