# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from data_designer.cli.services.review_service import ReviewService


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a sample parquet dataset for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to the test dataset
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [95, 88, 92, 85, 90],
        }
    )
    dataset_path = tmp_path / "test_dataset.parquet"
    df.to_parquet(dataset_path)
    return dataset_path


@pytest.fixture
def sample_csv_dataset(tmp_path: Path) -> Path:
    """Create a sample CSV dataset for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to the test CSV dataset
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    dataset_path = tmp_path / "test_dataset.csv"
    df.to_csv(dataset_path, index=False)
    return dataset_path


@pytest.fixture
def review_service(sample_dataset: Path) -> ReviewService:
    """Create a ReviewService instance.

    Args:
        sample_dataset: Path to sample dataset

    Returns:
        ReviewService instance
    """
    return ReviewService(sample_dataset)


def test_load_dataset(review_service: ReviewService) -> None:
    """Test loading dataset."""
    df = review_service.load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "name" in df.columns


def test_load_dataset_caching(review_service: ReviewService) -> None:
    """Test that dataset is cached after first load."""
    df1 = review_service.load_dataset()
    df2 = review_service.load_dataset()
    # Should return the same object (cached)
    assert df1 is df2


def test_load_csv_dataset(sample_csv_dataset: Path) -> None:
    """Test loading CSV dataset."""
    service = ReviewService(sample_csv_dataset)
    df = service.load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "col1" in df.columns


def test_get_dataset_info(review_service: ReviewService) -> None:
    """Test getting dataset metadata."""
    info = review_service.get_dataset_info()
    assert info["num_records"] == 5
    assert info["num_columns"] == 3
    assert "id" in info["columns"]
    assert "name" in info["columns"]
    assert "score" in info["columns"]


def test_get_record_by_index(review_service: ReviewService) -> None:
    """Test retrieving a record by index."""
    record = review_service.get_record_by_index(0)
    assert record["name"] == "Alice"
    assert record["score"] == 95


def test_get_record_by_index_last(review_service: ReviewService) -> None:
    """Test retrieving the last record."""
    record = review_service.get_record_by_index(4)
    assert record["name"] == "Eve"
    assert record["score"] == 90


def test_get_record_invalid_index_negative(review_service: ReviewService) -> None:
    """Test error handling for negative index."""
    with pytest.raises(ValueError, match="Invalid index -1"):
        review_service.get_record_by_index(-1)


def test_get_record_invalid_index_out_of_bounds(review_service: ReviewService) -> None:
    """Test error handling for out-of-bounds index."""
    with pytest.raises(ValueError, match="Invalid index 10"):
        review_service.get_record_by_index(10)


def test_submit_review(review_service: ReviewService) -> None:
    """Test submitting a review."""
    review_service.submit_review(
        record_index=0,
        rating="thumbs_up",
        comment="Great record!",
        reviewer="test_user",
    )

    # Verify review was saved
    reviews = review_service.get_reviews_dataframe()
    assert len(reviews) == 1
    assert reviews.iloc[0]["record_index"] == 0
    assert reviews.iloc[0]["rating"] == "thumbs_up"
    assert reviews.iloc[0]["comment"] == "Great record!"


def test_submit_review_invalid_rating(review_service: ReviewService) -> None:
    """Test error handling for invalid rating."""
    with pytest.raises(ValueError, match="Invalid rating 'invalid'"):
        review_service.submit_review(0, "invalid", "", "test")


def test_submit_review_invalid_index(review_service: ReviewService) -> None:
    """Test error handling for invalid record index."""
    with pytest.raises(ValueError, match="Invalid record_index 100"):
        review_service.submit_review(100, "thumbs_up", "", "test")


def test_get_review_progress_empty(review_service: ReviewService) -> None:
    """Test progress tracking with no reviews."""
    progress = review_service.get_review_progress()
    assert progress["total_records"] == 5
    assert progress["reviewed_records"] == 0
    assert progress["progress_percent"] == 0.0
    assert progress["reviewed_indices"] == set()


def test_get_review_progress_with_reviews(review_service: ReviewService) -> None:
    """Test progress tracking with reviews."""
    # Submit reviews for 2 records
    review_service.submit_review(0, "thumbs_up", "", "test")
    review_service.submit_review(2, "thumbs_down", "", "test")

    progress = review_service.get_review_progress()
    assert progress["total_records"] == 5
    assert progress["reviewed_records"] == 2
    assert progress["progress_percent"] == 40.0
    assert 0 in progress["reviewed_indices"]
    assert 2 in progress["reviewed_indices"]


def test_get_reviews_dataframe(review_service: ReviewService) -> None:
    """Test getting all reviews as DataFrame."""
    # Submit multiple reviews
    review_service.submit_review(0, "thumbs_up", "Good", "user1")
    review_service.submit_review(1, "thumbs_down", "Bad", "user2")

    reviews = review_service.get_reviews_dataframe()
    assert isinstance(reviews, pd.DataFrame)
    assert len(reviews) == 2
    assert list(reviews["record_index"]) == [0, 1]


def test_empty_dataset_progress(tmp_path: Path) -> None:
    """Test progress tracking with empty dataset."""
    empty_df = pd.DataFrame({"col1": []})
    dataset_path = tmp_path / "empty.parquet"
    empty_df.to_parquet(dataset_path)

    service = ReviewService(dataset_path)
    progress = service.get_review_progress()

    assert progress["total_records"] == 0
    assert progress["reviewed_records"] == 0
    assert progress["progress_percent"] == 0.0
