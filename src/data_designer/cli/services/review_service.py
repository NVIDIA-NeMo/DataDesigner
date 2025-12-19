# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from pathlib import Path

import pandas as pd

from data_designer.cli.repositories.review_repository import ReviewRecord, ReviewRepository


class ReviewService:
    """Service layer for dataset review operations.

    Provides business logic for reviewing datasets, including loading datasets,
    managing reviews, and tracking progress.
    """

    def __init__(self, dataset_path: Path):
        """Initialize service with dataset path.

        Args:
            dataset_path: Path to the dataset file to review
        """
        self.dataset_path = dataset_path
        self.repository = ReviewRepository(dataset_path)
        self._dataset: pd.DataFrame | None = None

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset based on file extension.

        Returns:
            DataFrame containing the dataset

        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If dataset file doesn't exist
        """
        if self._dataset is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

            ext = self.dataset_path.suffix.lower()
            if ext == ".parquet":
                self._dataset = pd.read_parquet(self.dataset_path, dtype_backend="pyarrow")
            elif ext == ".csv":
                self._dataset = pd.read_csv(self.dataset_path)
            elif ext == ".json":
                self._dataset = pd.read_json(self.dataset_path, lines=True)
            elif ext == ".jsonl":
                self._dataset = pd.read_json(self.dataset_path, lines=True)
            else:
                raise ValueError(
                    f"Unsupported file format: {ext}. Supported formats: .parquet, .csv, .json, .jsonl"
                )
        return self._dataset

    def get_dataset_info(self) -> dict[str, int | list[str]]:
        """Get metadata about the dataset.

        Returns:
            Dictionary with num_records, num_columns, and columns list
        """
        df = self.load_dataset()
        return {
            "num_records": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist(),
        }

    def get_record_by_index(self, index: int) -> pd.Series:
        """Get a single record by index.

        Args:
            index: Record index (0-based)

        Returns:
            Series containing the record data

        Raises:
            ValueError: If index is out of bounds
        """
        df = self.load_dataset()
        if index < 0 or index >= len(df):
            raise ValueError(f"Invalid index {index}. Dataset has {len(df)} records.")
        return df.iloc[index]

    def submit_review(self, record_index: int, rating: str, comment: str, reviewer: str = "default") -> None:
        """Submit a review for a record.

        Args:
            record_index: Index of the record being reviewed
            rating: Rating value ("thumbs_up" or "thumbs_down")
            comment: Optional comment text
            reviewer: Reviewer identifier

        Raises:
            ValueError: If rating is invalid or record_index is out of bounds
        """
        # Validate rating
        if rating not in ("thumbs_up", "thumbs_down"):
            raise ValueError(f"Invalid rating '{rating}'. Must be 'thumbs_up' or 'thumbs_down'.")

        # Validate record index
        df = self.load_dataset()
        if record_index < 0 or record_index >= len(df):
            raise ValueError(f"Invalid record_index {record_index}. Dataset has {len(df)} records.")

        # Create and save review
        review = ReviewRecord(
            record_index=record_index,
            timestamp=datetime.now().isoformat(),
            rating=rating,
            comment=comment,
            reviewer=reviewer,
        )
        self.repository.save_review(review)

    def get_review_progress(self) -> dict[str, int | float | set[int]]:
        """Get progress statistics.

        Returns:
            Dictionary with total_records, reviewed_records, progress_percent, and reviewed_indices
        """
        df = self.load_dataset()
        reviewed = self.repository.get_reviewed_indices()
        total = len(df)

        return {
            "total_records": total,
            "reviewed_records": len(reviewed),
            "progress_percent": (len(reviewed) / total * 100) if total > 0 else 0,
            "reviewed_indices": reviewed,
        }

    def get_reviews_dataframe(self) -> pd.DataFrame:
        """Get all reviews as a DataFrame.

        Returns:
            DataFrame containing all reviews
        """
        return self.repository.load_reviews()
