# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading

import pytest

from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker


@pytest.fixture
def tracker() -> ProgressTracker:
    return ProgressTracker(total_records=100, label="test column 'name'")


class TestProgressTrackerInit:
    def test_initializes_with_correct_values(self) -> None:
        tracker = ProgressTracker(total_records=100, label="test label")

        assert tracker.total_records == 100
        assert tracker.label == "test label"
        assert tracker.completed == 0
        assert tracker.success == 0
        assert tracker.failed == 0

    def test_calculates_log_interval_from_percentage(self) -> None:
        tracker = ProgressTracker(total_records=100, label="test", log_interval_percent=10)
        assert tracker.log_interval == 10

        tracker = ProgressTracker(total_records=100, label="test", log_interval_percent=25)
        assert tracker.log_interval == 25

        tracker = ProgressTracker(total_records=1000, label="test", log_interval_percent=5)
        assert tracker.log_interval == 50

    def test_log_interval_minimum_is_one(self) -> None:
        tracker = ProgressTracker(total_records=5, label="test", log_interval_percent=10)
        assert tracker.log_interval >= 1

    def test_handles_zero_total_records(self) -> None:
        tracker = ProgressTracker(total_records=0, label="test")
        assert tracker.log_interval == 1
        assert tracker.total_records == 0


class TestRecordSuccess:
    def test_increments_completed_and_success(self, tracker: ProgressTracker) -> None:
        tracker.record_success()

        assert tracker.completed == 1
        assert tracker.success == 1
        assert tracker.failed == 0

    def test_multiple_successes(self, tracker: ProgressTracker) -> None:
        for _ in range(5):
            tracker.record_success()

        assert tracker.completed == 5
        assert tracker.success == 5
        assert tracker.failed == 0


class TestRecordFailure:
    def test_increments_completed_and_failed(self, tracker: ProgressTracker) -> None:
        tracker.record_failure()

        assert tracker.completed == 1
        assert tracker.success == 0
        assert tracker.failed == 1

    def test_multiple_failures(self, tracker: ProgressTracker) -> None:
        for _ in range(5):
            tracker.record_failure()

        assert tracker.completed == 5
        assert tracker.success == 0
        assert tracker.failed == 5


class TestMixedRecording:
    def test_tracks_mixed_successes_and_failures(self, tracker: ProgressTracker) -> None:
        tracker.record_success()
        tracker.record_success()
        tracker.record_failure()
        tracker.record_success()
        tracker.record_failure()

        assert tracker.completed == 5
        assert tracker.success == 3
        assert tracker.failed == 2


class TestLogging:
    def test_log_start_logs_worker_info(self, tracker: ProgressTracker, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            tracker.log_start(max_workers=8)

        assert "8 concurrent workers" in caplog.text
        assert "test column 'name'" in caplog.text

    def test_logs_progress_at_interval(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total_records=10, label="test", log_interval_percent=50)

        with caplog.at_level(logging.INFO):
            for _ in range(5):
                tracker.record_success()

        assert "5/10" in caplog.text
        assert "50%" in caplog.text

    def test_log_final_logs_remaining_progress(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total_records=10, label="test", log_interval_percent=50)

        for _ in range(3):
            tracker.record_success()

        with caplog.at_level(logging.INFO):
            tracker.log_final()

        assert "3/10" in caplog.text

    def test_progress_log_includes_rate_and_eta(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total_records=10, label="test", log_interval_percent=50)

        with caplog.at_level(logging.INFO):
            for _ in range(5):
                tracker.record_success()

        assert "rec/s" in caplog.text
        assert "eta" in caplog.text

    def test_progress_log_shows_ok_and_failed_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total_records=10, label="test", log_interval_percent=50)

        with caplog.at_level(logging.INFO):
            for _ in range(3):
                tracker.record_success()
            for _ in range(2):
                tracker.record_failure()

        assert "3 ok" in caplog.text
        assert "2 failed" in caplog.text


class TestThreadSafety:
    def test_concurrent_record_calls_are_thread_safe(self) -> None:
        tracker = ProgressTracker(total_records=1000, label="test", log_interval_percent=100)
        num_threads = 10
        records_per_thread = 100

        def record_many_successes() -> None:
            for _ in range(records_per_thread):
                tracker.record_success()

        def record_many_failures() -> None:
            for _ in range(records_per_thread):
                tracker.record_failure()

        threads = []
        for i in range(num_threads):
            if i % 2 == 0:
                thread = threading.Thread(target=record_many_successes)
            else:
                thread = threading.Thread(target=record_many_failures)
            threads.append(thread)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        expected_success = (num_threads // 2) * records_per_thread
        expected_failed = (num_threads - num_threads // 2) * records_per_thread

        assert tracker.completed == num_threads * records_per_thread
        assert tracker.success == expected_success
        assert tracker.failed == expected_failed


class TestEdgeCases:
    def test_handles_very_small_total_records(self) -> None:
        tracker = ProgressTracker(total_records=1, label="test")

        tracker.record_success()

        assert tracker.completed == 1
        assert tracker.success == 1

    def test_handles_log_interval_larger_than_total(self) -> None:
        tracker = ProgressTracker(total_records=5, label="test", log_interval_percent=50)

        for _ in range(5):
            tracker.record_success()

        assert tracker.completed == 5

    def test_log_final_handles_zero_records_processed(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = ProgressTracker(total_records=10, label="test")

        with caplog.at_level(logging.INFO):
            tracker.log_final()

        # Should not raise, may or may not log depending on implementation

    def test_progress_percentage_with_zero_total(self) -> None:
        tracker = ProgressTracker(total_records=0, label="test")

        # Should not raise division by zero
        tracker.record_success()

        assert tracker.completed == 1
