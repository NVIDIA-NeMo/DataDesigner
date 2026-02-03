# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from threading import Lock

from data_designer.logging import RandomEmoji

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Thread-safe progress tracker for monitoring concurrent task completion.

    Tracks completed, successful, and failed task counts and logs progress
    at configurable intervals. Designed for use with ConcurrentThreadExecutor
    to provide visibility into long-running batch operations.

    Example usage:
        tracker = ProgressTracker(total_records=100, label="LLM_TEXT column 'response'")
        tracker.log_start(max_workers=8)

        # In callbacks from ConcurrentThreadExecutor:
        tracker.record_success()  # or tracker.record_failure()

        # After executor completes:
        tracker.log_final()
    """

    def __init__(self, total_records: int, label: str, log_interval_percent: int = 10):
        """
        Initialize the progress tracker.

        Args:
            total_records: Total number of records to process.
            label: Human-readable label for log messages (e.g., "LLM_TEXT column 'response'").
            log_interval_percent: How often to log progress as a percentage (default 10%).
        """
        self.total_records = total_records
        self.label = label

        self.completed = 0
        self.success = 0
        self.failed = 0

        interval_fraction = max(1, log_interval_percent) / 100.0
        self.log_interval = max(1, int(total_records * interval_fraction)) if total_records > 0 else 1
        self.next_log_at = self.log_interval

        self.start_time = time.perf_counter()
        self.lock = Lock()
        self._random_emoji = RandomEmoji()

    def log_start(self, max_workers: int) -> None:
        """Log the start of processing with worker count and interval information."""
        logger.info(
            "🐙 Processing %s with %d concurrent workers",
            self.label,
            max_workers,
        )
        logger.info(
            "🧭 %s will report progress every %d record(s).",
            self.label,
            self.log_interval,
        )

    def record_success(self) -> None:
        """Record a successful task completion and log progress if at interval."""
        self._record_completion(success=True)

    def record_failure(self) -> None:
        """Record a failed task completion and log progress if at interval."""
        self._record_completion(success=False)

    def log_final(self) -> None:
        """Log final progress summary."""
        with self.lock:
            if self.completed > 0:
                self._log_progress_unlocked()

    def _record_completion(self, *, success: bool) -> None:
        should_log = False
        with self.lock:
            self.completed += 1
            if success:
                self.success += 1
            else:
                self.failed += 1

            if self.completed >= self.next_log_at and self.completed < self.total_records:
                should_log = True
                while self.next_log_at <= self.completed:
                    self.next_log_at += self.log_interval

        if should_log:
            with self.lock:
                self._log_progress_unlocked()

    def _log_progress_unlocked(self) -> None:
        """Log current progress. Must be called while holding the lock."""
        elapsed = time.perf_counter() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total_records - self.completed)
        eta = f"{(remaining / rate):.1f}s" if rate > 0 else "unknown"
        percent = (self.completed / self.total_records) * 100 if self.total_records else 100.0

        logger.info(
            "  |-- %s %s progress: %d/%d (%.0f%%) complete, %d ok, %d failed, %.2f rec/s, eta %s",
            self._random_emoji.progress(percent),
            self.label,
            self.completed,
            self.total_records,
            percent,
            self.success,
            self.failed,
            rate,
            eta,
        )
