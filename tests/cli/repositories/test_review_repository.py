# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data_designer.cli.repositories.review_repository import ReviewRecord, ReviewRepository


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a sample dataset for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to the test dataset
    """
    df = pd.DataFrame({"col1": [1, 2, 3]})
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def repository(sample_dataset: Path) -> ReviewRepository:
    """Create a ReviewRepository instance.

    Args:
        sample_dataset: Path to sample dataset

    Returns:
        ReviewRepository instance
    """
    return ReviewRepository(sample_dataset)


def test_get_review_file_path(repository: ReviewRepository) -> None:
    """Test review file path generation."""
    assert repository.review_file_path.name == "test_reviews.csv"
    assert repository.review_file_path.parent == repository.dataset_path.parent


def test_load_reviews_empty(repository: ReviewRepository) -> None:
    """Test loading reviews when no reviews exist."""
    reviews = repository.load_reviews()
    assert isinstance(reviews, pd.DataFrame)
    assert len(reviews) == 0
    assert list(reviews.columns) == ["record_index", "timestamp", "rating", "comment", "reviewer"]


def test_save_and_load_review(repository: ReviewRepository) -> None:
    """Test saving and loading a review."""
    review = ReviewRecord(
        record_index=0,
        timestamp=datetime.now().isoformat(),
        rating="thumbs_up",
        comment="Test comment",
        reviewer="test_user",
    )

    repository.save_review(review)
    reviews = repository.load_reviews()

    assert len(reviews) == 1
    assert reviews.iloc[0]["record_index"] == 0
    assert reviews.iloc[0]["rating"] == "thumbs_up"
    assert reviews.iloc[0]["comment"] == "Test comment"
    assert reviews.iloc[0]["reviewer"] == "test_user"


def test_save_multiple_reviews(repository: ReviewRepository) -> None:
    """Test saving multiple reviews."""
    review1 = ReviewRecord(0, datetime.now().isoformat(), "thumbs_up", "Good", "user1")
    review2 = ReviewRecord(1, datetime.now().isoformat(), "thumbs_down", "Bad", "user1")

    repository.save_review(review1)
    repository.save_review(review2)

    reviews = repository.load_reviews()
    assert len(reviews) == 2


def test_get_reviewed_indices(repository: ReviewRepository) -> None:
    """Test getting reviewed indices."""
    review1 = ReviewRecord(0, datetime.now().isoformat(), "thumbs_up", "", "user")
    review2 = ReviewRecord(2, datetime.now().isoformat(), "thumbs_down", "", "user")

    repository.save_review(review1)
    repository.save_review(review2)

    indices = repository.get_reviewed_indices()
    assert indices == {0, 2}


def test_get_reviewed_indices_empty(repository: ReviewRepository) -> None:
    """Test getting reviewed indices when no reviews exist."""
    indices = repository.get_reviewed_indices()
    assert indices == set()


def test_multiple_reviews_same_record(repository: ReviewRepository) -> None:
    """Test multiple reviews for the same record (append behavior)."""
    review1 = ReviewRecord(0, datetime.now().isoformat(), "thumbs_up", "First review", "user1")
    review2 = ReviewRecord(0, datetime.now().isoformat(), "thumbs_down", "Second review", "user2")

    repository.save_review(review1)
    repository.save_review(review2)

    reviews = repository.load_reviews()
    assert len(reviews) == 2

    # Both reviews should exist
    record_0_reviews = reviews[reviews["record_index"] == 0]
    assert len(record_0_reviews) == 2


def test_export_reviews(repository: ReviewRepository, tmp_path: Path) -> None:
    """Test exporting reviews to a custom path."""
    review = ReviewRecord(0, datetime.now().isoformat(), "thumbs_up", "Export test", "user")
    repository.save_review(review)

    export_path = tmp_path / "exported_reviews.csv"
    result_path = repository.export_reviews(export_path)

    assert result_path == export_path
    assert export_path.exists()

    # Verify exported content
    exported_df = pd.read_csv(export_path)
    assert len(exported_df) == 1
    assert exported_df.iloc[0]["comment"] == "Export test"
