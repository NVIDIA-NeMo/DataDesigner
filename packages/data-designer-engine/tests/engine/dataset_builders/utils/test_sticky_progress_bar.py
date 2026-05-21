# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
import os
import re
import shutil
from unittest.mock import patch

import pytest

from data_designer.engine.dataset_builders.utils.async_progress_reporter import AsyncProgressReporter
from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.engine.dataset_builders.utils.sticky_progress_bar import StickyProgressBar

CURSOR_UP_CLEAR = "\033[A\033[2K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
_ALL_ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")


class FakeTTY(io.StringIO):
    """StringIO that reports itself as a TTY so StickyProgressBar activates."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def tty_stream() -> FakeTTY:
    return FakeTTY()


def _clean(text: str) -> str:
    return _ALL_ANSI_RE.sub("", text).replace("\r", "")


def _last_panel_lines(output: str) -> list[str]:
    clean = _clean(output)
    panel_start = clean.rfind("╭")
    assert panel_start >= 0
    return clean[panel_start:].splitlines()


def test_no_output_when_not_tty() -> None:
    stream = io.StringIO()
    with StickyProgressBar(stream=stream) as bar:
        bar.add_bar("a", "col_a", 10)
        bar.update("a", completed=5, success=5)
    assert stream.getvalue() == ""


def test_hides_and_shows_cursor(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream):
        pass
    output = tty_stream.getvalue()
    assert output.startswith(HIDE_CURSOR)
    assert output.endswith(SHOW_CURSOR)


def test_renders_bounded_throughput_panel(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "column 'a'", 100)
        bar.add_bar("b", "column 'b'", 100)
        bar.update_many({"a": (10, 10, 0), "b": (20, 20, 0)})

        assert bar.drawn_lines == 16
        panel = "\n".join(_last_panel_lines(tty_stream.getvalue()))
        assert "Throughput" in panel
        assert "rec/s" in panel
        assert "column 'a': 10/100" in panel
        assert "column 'b': 20/100" in panel
        assert "╭" in panel
        assert "╰" in panel


def test_panel_height_stable_across_many_updates(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.add_bar("b", "col_b", 100)

        for i in range(50):
            bar.update_many({"a": (i, i, 0), "b": (i * 2, i * 2, 0)})

        snapshot = tty_stream.getvalue()
        bar.update("a", completed=50, success=50)

        assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 16
        assert bar.drawn_lines == 16


def test_log_interleaving_preserves_panel_height(tty_stream: FakeTTY) -> None:
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(tty_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    try:
        with StickyProgressBar(stream=tty_stream) as bar:
            bar.add_bar("x", "col_x", 100)
            bar.add_bar("y", "col_y", 100)

            for i in range(10):
                bar.update("x", completed=i, success=i)
                root_logger.info("log at step %d", i)
                bar.update("y", completed=i, success=i)

            snapshot = tty_stream.getvalue()
            bar.update("x", completed=20, success=20)
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 16
    finally:
        root_logger.removeHandler(handler)


def test_narrow_terminal_keeps_panel_within_width(tty_stream: FakeTTY) -> None:
    narrow = os.terminal_size((36, 24))
    with patch.object(shutil, "get_terminal_size", return_value=narrow):
        with StickyProgressBar(stream=tty_stream) as bar:
            bar.add_bar("a", "column 'verification_1'", 300)
            bar.update("a", completed=50, success=50)

            output = tty_stream.getvalue()
            for line in _last_panel_lines(output):
                assert len(line) <= 35


def test_update_many_single_redraw(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.add_bar("b", "col_b", 100)
        before = tty_stream.getvalue()

        bar.update_many({"a": (10, 10, 0), "b": (20, 20, 0)})
        after = tty_stream.getvalue()

        new_output = after[len(before) :]
        assert new_output.count(CURSOR_UP_CLEAR) == 16

        clean = _clean(after)
        assert "10/100" in clean
        assert "20/100" in clean


def test_update_many_includes_failures_and_skips(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.update_many({"a": (10, 7, 2, 1), "unknown": (5, 5, 0, 0)})

        clean = _clean(tty_stream.getvalue())
        assert "10/100" in clean
        assert "2 failed" in clean
        assert "1 skipped" in clean
        assert "unknown" not in clean


def test_reporter_updates_and_logs_keep_drawn_lines_in_sync(tty_stream: FakeTTY) -> None:
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(tty_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    try:
        bar = StickyProgressBar(stream=tty_stream)
        trackers = {
            "col_a": ProgressTracker(total_records=100, label="column 'a'", quiet=True),
            "col_b": ProgressTracker(total_records=100, label="column 'b'", quiet=True),
            "col_c": ProgressTracker(total_records=100, label="column 'c'", quiet=True),
        }

        with bar:
            reporter = AsyncProgressReporter(trackers, report_interval=0.1, progress_bar=bar)
            reporter.log_start(num_row_groups=1)

            snapshot = tty_stream.getvalue()
            reporter.record_success("col_a")
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 16

            for i in range(49):
                if i % 10 == 0:
                    root_logger.info("Processing batch %d", i)
                reporter.record_success("col_b")
                reporter.record_skipped("col_c")

            snapshot = tty_stream.getvalue()
            reporter.log_final()
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 16
            assert bar.drawn_lines == 16
    finally:
        root_logger.removeHandler(handler)
