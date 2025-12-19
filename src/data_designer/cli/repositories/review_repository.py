# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ReviewRecord:
    """Single review feedback record."""

    record_index: int
    timestamp: str
    rating: str  # "thumbs_up" or "thumbs_down"
    comment: str
    reviewer: str = "default"


class ReviewRepository:
    """Repository for review feedback persistence.

    Handles saving and loading review feedback to/from CSV files.
    Reviews are stored as {dataset_stem}_reviews.csv in artifacts/reviews/ directory.
    """

    def __init__(self, dataset_path: Path):
        """Initialize repository with dataset path.

        Args:
            dataset_path: Path to the dataset file being reviewed
        """
        self.dataset_path = dataset_path
        self.review_file_path = self._get_review_file_path()

    def _get_review_file_path(self) -> Path:
        """Get review CSV path in artifacts/reviews directory.

        Returns:
            Path to the reviews CSV file
        """
        # Find the artifacts directory by going up from dataset path
        current = self.dataset_path.parent
        artifacts_dir = None

        # Search up to 5 levels up for 'artifacts' directory
        for _ in range(5):
            if current.name == "artifacts":
                artifacts_dir = current
                break
            if current.parent == current:  # Reached root
                break
            current = current.parent

        # If artifacts directory not found, create it relative to dataset
        if artifacts_dir is None:
            artifacts_dir = self.dataset_path.parent / "artifacts"

        # Create reviews subdirectory under artifacts
        reviews_dir = artifacts_dir / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)

        return reviews_dir / f"{self.dataset_path.stem}_reviews.csv"

    def load_reviews(self) -> pd.DataFrame:
        """Load existing reviews or return empty DataFrame.

        Returns:
            DataFrame with columns: record_index, timestamp, rating, comment, reviewer
        """
        if self.review_file_path.exists():
            return pd.read_csv(self.review_file_path)
        return pd.DataFrame(columns=["record_index", "timestamp", "rating", "comment", "reviewer"])

    def save_review(self, review: ReviewRecord) -> None:
        """Append a single review to the CSV file.

        Args:
            review: ReviewRecord to save
        """
        file_exists = self.review_file_path.exists()

        # Ensure parent directory exists
        self.review_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to CSV
        with open(self.review_file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["record_index", "timestamp", "rating", "comment", "reviewer"])

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write review
            writer.writerow(
                {
                    "record_index": review.record_index,
                    "timestamp": review.timestamp,
                    "rating": review.rating,
                    "comment": review.comment,
                    "reviewer": review.reviewer,
                }
            )

    def get_reviewed_indices(self) -> set[int]:
        """Get set of record indices that have been reviewed.

        Returns:
            Set of record indices (0-based) that have reviews
        """
        reviews = self.load_reviews()
        if reviews.empty:
            return set()
        return set(reviews["record_index"].unique())

    def export_reviews(self, output_path: Path | None = None) -> Path:
        """Export reviews to specified path or default location.

        Args:
            output_path: Optional custom export path

        Returns:
            Path where reviews were exported
        """
        export_path = output_path or self.review_file_path
        reviews = self.load_reviews()
        reviews.to_csv(export_path, index=False)
        return export_path
