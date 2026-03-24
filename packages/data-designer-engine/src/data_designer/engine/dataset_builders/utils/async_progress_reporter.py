# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time

from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.logging import LOG_INDENT

logger = logging.getLogger(__name__)

DEFAULT_REPORT_INTERVAL = 5.0


class AsyncProgressReporter:
    """Consolidated progress reporter for async generation.

    Owns per-column ProgressTracker instances (in quiet mode) and emits
    a single grouped log block at most once per ``report_interval`` seconds.
    """

    def __init__(
        self,
        trackers: dict[str, ProgressTracker],
        *,
        report_interval: float = DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._trackers = trackers
        self._report_interval = report_interval
        self._start_time = time.perf_counter()
        self._last_report_time: float = self._start_time
        self._last_reported_total: int = -1

    def log_start(self, num_row_groups: int) -> None:
        cols = ", ".join(self._trackers)
        total = sum(t.total_records for t in self._trackers.values())
        logger.info(
            "⚡️ Async generation: %d column(s) (%s), %d tasks across %d row group(s)",
            len(self._trackers),
            cols,
            total,
            num_row_groups,
        )

    def record_success(self, column: str) -> None:
        if tracker := self._trackers.get(column):
            tracker.record_success()
            self._maybe_report()

    def record_failure(self, column: str) -> None:
        if tracker := self._trackers.get(column):
            tracker.record_failure()
            self._maybe_report()

    def log_final(self) -> None:
        self._emit()
        elapsed = time.perf_counter() - self._start_time
        total_ok = sum(t.success for t in self._trackers.values())
        total_fail = sum(t.failed for t in self._trackers.values())
        logger.info(
            "✅ Async generation complete [%.1fs]: %d ok, %d failed across %d column(s)",
            elapsed,
            total_ok,
            total_fail,
            len(self._trackers),
        )

    def _maybe_report(self) -> None:
        now = time.perf_counter()
        if now - self._last_report_time < self._report_interval:
            return
        self._last_report_time = now
        self._emit()

    def _emit(self) -> None:
        current_total = sum(t.completed for t in self._trackers.values())
        if current_total == self._last_reported_total:
            return
        self._last_reported_total = current_total

        elapsed = time.perf_counter() - self._start_time
        logger.info("📊 Progress [%.1fs]:", elapsed)
        for col, tracker in self._trackers.items():
            with tracker.lock:
                pct = (tracker.completed / tracker.total_records * 100) if tracker.total_records else 100.0
                rate = tracker.completed / elapsed if elapsed > 0 else 0.0
                emoji = tracker._random_emoji.progress(pct)
            logger.info(
                "%s%s %s: %d/%d (%.0f%%) %.1f rec/s",
                LOG_INDENT,
                emoji,
                col,
                tracker.completed,
                tracker.total_records,
                pct,
                rate,
            )
