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
from data_designer.engine.dataset_builders.utils.sticky_progress_bar import (
    StickyProgressBar,
)

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


def test_drawn_lines_tracks_add_and_remove(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 10)
        bar.add_bar("b", "col_b", 10)
        bar.add_bar("c", "col_c", 10)
        assert bar.drawn_lines == 3

        bar.remove_bar("a")
        assert bar.drawn_lines == 2

        bar.add_bar("d", "col_d", 10)
        assert bar.drawn_lines == 3

        bar.update("b", completed=5, success=5)
        assert bar.drawn_lines == 3

        bar.remove_bar("b")
        bar.remove_bar("c")
        bar.remove_bar("d")
        assert bar.drawn_lines == 0


def test_drawn_lines_stable_across_many_updates(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.add_bar("b", "col_b", 100)
        bar.add_bar("c", "col_c", 100)
        for i in range(50):
            bar.update("a", completed=i, success=i)
            bar.update("b", completed=i, success=i)
            bar.update("c", completed=i, success=i)

        snapshot = tty_stream.getvalue()
        bar.update("a", completed=50, success=50)
        assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 3


def test_log_interleaving_preserves_drawn_lines(tty_stream: FakeTTY) -> None:
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(tty_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    try:
        with StickyProgressBar(stream=tty_stream) as bar:
            bar.add_bar("x", "col_x", 100)
            bar.add_bar("y", "col_y", 100)
            bar.add_bar("z", "col_z", 100)

            for i in range(20):
                bar.update("x", completed=i, success=i)
                root_logger.info("log at step %d", i)
                bar.update("y", completed=i, success=i)
                bar.update("z", completed=i, success=i)

            snapshot = tty_stream.getvalue()
            bar.update("x", completed=20, success=20)
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 3
    finally:
        root_logger.removeHandler(handler)


def test_wrapping_counts_physical_lines(tty_stream: FakeTTY) -> None:
    narrow = os.terminal_size((40, 24))
    with patch.object(shutil, "get_terminal_size", return_value=narrow):
        with StickyProgressBar(stream=tty_stream) as bar:
            bar.add_bar("a", "col_a", 100)
            bar.add_bar("b", "col_b", 100)

            original_format = bar._format_bar

            def oversized_format(b: object, width: int, label_width: int) -> str:
                line = original_format(b, width, label_width)
                return line + "X" * 20

            with patch.object(bar, "_format_bar", side_effect=oversized_format):
                bar.update("a", completed=5, success=5)

            snapshot = tty_stream.getvalue()
            bar.update("b", completed=1, success=1)
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) > 2


def test_wrapping_stable_across_updates(tty_stream: FakeTTY) -> None:
    narrow = os.terminal_size((40, 24))
    with patch.object(shutil, "get_terminal_size", return_value=narrow):
        with StickyProgressBar(stream=tty_stream) as bar:
            bar.add_bar("a", "col_a", 100)
            bar.add_bar("b", "col_b", 100)

            snapshot = tty_stream.getvalue()
            bar.update("a", completed=0, success=0)
            initial_clears = tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR)

            for i in range(1, 21):
                bar.update("a", completed=i, success=i)
                bar.update("b", completed=i, success=i)

            snapshot = tty_stream.getvalue()
            bar.update("a", completed=21, success=21)
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == initial_clears


def test_narrow_terminal_graceful_degradation(tty_stream: FakeTTY) -> None:
    narrow = os.terminal_size((30, 24))
    with patch.object(shutil, "get_terminal_size", return_value=narrow):
        with StickyProgressBar(stream=tty_stream) as bar:
            bar.add_bar("a", "column 'verification_1'", 300)
            bar.update("a", completed=50, success=50)

            snapshot = tty_stream.getvalue()
            bar.update("a", completed=51, success=51)
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 1

            output = tty_stream.getvalue()
            clean = _ALL_ANSI_RE.sub("", output).replace("\r", "")
            for line in clean.split("\n"):
                assert len(line) <= 29


def test_update_many_single_redraw(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.add_bar("b", "col_b", 100)
        before = tty_stream.getvalue()

        bar.update_many({"a": (10, 10, 0), "b": (20, 20, 0)})
        after = tty_stream.getvalue()

        new_output = after[len(before) :]
        assert new_output.count(CURSOR_UP_CLEAR) == 2

        clean = _ALL_ANSI_RE.sub("", after)
        assert "10/100" in clean
        assert "20/100" in clean


def test_update_many_ignores_unknown_keys(tty_stream: FakeTTY) -> None:
    with StickyProgressBar(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.update_many({"a": (10, 10, 0), "unknown": (5, 5, 0)})

        clean = _ALL_ANSI_RE.sub("", tty_stream.getvalue())
        assert "10/100" in clean
        assert "unknown" not in clean

        snapshot = tty_stream.getvalue()
        bar.update("a", completed=11, success=11)
        assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 1


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
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 3

            for i in range(49):
                if i % 10 == 0:
                    root_logger.info("Processing batch %d", i)
                reporter.record_success("col_b")
                reporter.record_success("col_c")

            snapshot = tty_stream.getvalue()
            reporter.record_success("col_a")
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 3

            reporter.log_final()
            assert bar.drawn_lines == 0
    finally:
        root_logger.removeHandler(handler)
