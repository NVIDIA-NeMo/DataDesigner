# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.logging import LOG_INDENT

if TYPE_CHECKING:
    from data_designer.engine.dataset_builders.utils.sticky_progress_bar import StickyProgressBar

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
        progress_bar: StickyProgressBar | None = None,
    ) -> None:
        self._trackers = trackers
        self._report_interval = report_interval
        self._start_time = time.perf_counter()
        self._last_report_time: float = self._start_time
        self._last_reported_total: int = -1
        self._bar = progress_bar
        if self._bar is not None:
            for col, tracker in trackers.items():
                self._bar.add_bar(col, f"column '{col}'", tracker.total_records)

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

    def record_skipped(self, column: str) -> None:
        if tracker := self._trackers.get(column):
            tracker.record_skipped()
            self._maybe_report()

    def log_final(self) -> None:
        if self._bar is not None and self._bar.is_active:
            for col in self._trackers:
                self._bar.remove_bar(col)
        else:
            self._emit()
        elapsed = time.perf_counter() - self._start_time
        snapshots = [tracker.get_snapshot(elapsed) for tracker in self._trackers.values()]
        total_ok = sum(snapshot[2] for snapshot in snapshots)
        total_fail = sum(snapshot[3] for snapshot in snapshots)
        total_skipped = sum(snapshot[4] for snapshot in snapshots)
        skipped_suffix = f", {total_skipped} skipped" if total_skipped else ""
        logger.info(
            "✅ Async generation complete [%.1fs]: %d ok, %d failed%s across %d column(s)",
            elapsed,
            total_ok,
            total_fail,
            skipped_suffix,
            len(self._trackers),
        )

    def _maybe_report(self) -> None:
        if self._bar is not None and self._bar.is_active:
            self._update_bar()
            return
        now = time.perf_counter()
        if now - self._last_report_time < self._report_interval:
            return
        self._last_report_time = now
        self._emit()

    def _update_bar(self) -> None:
        elapsed = time.perf_counter() - self._start_time
        for col, tracker in self._trackers.items():
            completed, _total, success, failed, _skipped, _pct, _rate, _emoji = tracker.get_snapshot(elapsed)
            self._bar.update(col, completed=completed, success=success, failed=failed)

    def _emit(self) -> None:
        current_total = sum(tracker.get_snapshot(0.0)[0] for tracker in self._trackers.values())
        if current_total == self._last_reported_total:
            return
        self._last_reported_total = current_total

        elapsed = time.perf_counter() - self._start_time
        logger.info("📊 Progress [%.1fs]:", elapsed)
        for col, tracker in self._trackers.items():
            completed, total_records, _success, _failed, skipped, pct, rate, emoji = tracker.get_snapshot(elapsed)
            skipped_suffix = f", {skipped} skipped" if skipped else ""
            logger.info(
                "%s%s %s: %d/%d (%.0f%%) %.1f rec/s%s",
                LOG_INDENT,
                emoji,
                col,
                completed,
                total_records,
                pct,
                rate,
                skipped_suffix,
            )
