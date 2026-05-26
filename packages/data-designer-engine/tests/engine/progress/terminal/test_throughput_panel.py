# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import time
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from data_designer.engine.models.usage_events import TokenUsageEvent, emit_token_usage_event
from data_designer.engine.observability import (
    RequestAdmissionEvent,
    RuntimeCorrelation,
    emit_request_admission_event,
)
from data_designer.engine.progress.reporter import AsyncProgressReporter
from data_designer.engine.progress.terminal.throughput_panel import (
    _CHART_LINE_COUNT,
    _MAX_RATE_SAMPLES,
    _RATE_SAMPLE_INTERVAL_SECONDS,
    TerminalThroughputPanel,
    _BarState,
    _fit_series,
)
from data_designer.engine.progress.tracker import ProgressTracker

CURSOR_UP_CLEAR = "\033[A\033[2K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
_ALL_ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")


class FakeTTY(io.StringIO):
    """StringIO that reports itself as a TTY so TerminalThroughputPanel activates."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def tty_stream() -> FakeTTY:
    return FakeTTY()


@pytest.fixture(autouse=True)
def fixed_terminal_size() -> Iterator[None]:
    with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
        yield


def _clean(text: str) -> str:
    return _ALL_ANSI_RE.sub("", text).replace("\r", "")


def _correlation(run_id: str) -> RuntimeCorrelation:
    return RuntimeCorrelation(
        run_id=run_id,
        row_group=0,
        task_column="col_a",
        task_type="cell",
        scheduling_group_kind="model",
        scheduling_group_identity_hash="hash",
        task_execution_id="task-exec",
    )


def _last_panel_lines(output: str) -> list[str]:
    clean = _clean(output)
    panel_start = clean.rfind("\n╭")
    panel_start = panel_start + 1 if panel_start >= 0 else clean.rfind("╭")
    assert panel_start >= 0
    return clean[panel_start:].splitlines()


def _chart_lines(panel_lines: list[str]) -> list[str]:
    separator_index = next(index for index, line in enumerate(panel_lines) if "├" in line)
    return panel_lines[2:separator_index]


def _marker_positions(panel_lines: list[str]) -> list[tuple[int, int]]:
    return [(row_index, line.index("◆")) for row_index, line in enumerate(_chart_lines(panel_lines)) if "◆" in line]


def test_no_output_when_not_tty() -> None:
    stream = io.StringIO()
    with TerminalThroughputPanel(stream=stream) as bar:
        bar.add_bar("a", "col_a", 10)
        bar.update("a", completed=5, success=5)
    assert stream.getvalue() == ""


def test_hides_and_shows_cursor(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream):
        pass
    output = tty_stream.getvalue()
    assert output.startswith(HIDE_CURSOR)
    assert output.endswith(SHOW_CURSOR)


def test_tiny_terminal_falls_back_to_no_panel(tty_stream: FakeTTY) -> None:
    with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((20, 24))):
        with TerminalThroughputPanel(stream=tty_stream) as bar:
            assert bar.is_active is False
            bar.add_bar("a", "col_a", 10)
            bar.update("a", completed=5, success=5, force=True)

    assert tty_stream.getvalue() == ""


def test_renders_bounded_throughput_panel(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "column 'a'", 100)
        bar.add_bar("b", "column 'b'", 100)
        bar.update_many({"a": (10, 10, 0, 0), "b": (20, 20, 0, 0)}, force=True)

        assert bar.drawn_lines == 22
        panel_lines = _last_panel_lines(tty_stream.getvalue())
        panel = "\n".join(panel_lines)
        assert "Throughput" in panel
        assert "rec/s" in panel
        assert "now rec/s" in panel
        assert "avg rec/s" in panel
        assert "column 'a'" in panel
        assert "10/100" in panel
        assert "column 'b'" in panel
        assert "20/100" in panel
        header = next(line for line in panel_lines if "now rec/s" in line)
        row = next(line for line in panel_lines if "column 'a'" in line)
        assert "|" not in header
        assert "|" not in row
        assert "in tok/s" not in panel
        assert "out tok/s" not in panel
        assert header.index("avg rec/s") < header.index("done")
        assert "━" in row
        assert row.rindex("0.0") < row.index("10/100")
        assert row.index("10/100") < row.index("━")
        assert "╭" in panel
        assert "╰" in panel


def test_model_usage_rates_render_in_separate_table(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "column 'a'", 100)
        bar.update("a", completed=10, success=10, force=True)
        bar._start_time = time.perf_counter() - 10.0  # noqa: SLF001
        bar.record_model_usage(
            model_alias="test",
            model_name="test-model",
            input_tokens=100,
            output_tokens=25,
            force=True,
        )

        panel = "\n".join(_last_panel_lines(tty_stream.getvalue()))
        assert "model alias" in panel
        assert "model name" in panel
        assert "test" in panel
        assert "test-model" in panel
        assert "rpm" in panel
        assert "in tok/s" in panel
        assert "out tok/s" in panel
        assert "6.0" in panel
        assert "10.0" in panel
        assert "2.5" in panel


def test_many_columns_and_models_do_not_shrink_chart(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        for index in range(8):
            bar.add_bar(f"col_{index}", f"column_{index}", 100)
        bar.update_many(
            {f"col_{index}": (index + 1, index + 1, 0, 0) for index in range(8)},
            force=True,
        )
        for index in range(8):
            bar.record_model_usage(
                model_alias=f"model_{index}",
                model_name=f"provider/model-{index}",
                input_tokens=100 + index,
                output_tokens=10 + index,
                force=True,
            )

        panel_lines = _last_panel_lines(tty_stream.getvalue())
        panel = "\n".join(panel_lines)
        assert len(_chart_lines(panel_lines)) == _CHART_LINE_COUNT
        assert len(panel_lines) > 22
        assert "more column(s)" not in panel
        assert "more model(s)" not in panel
        for index in range(8):
            assert f"column_{index}" in panel
            assert f"model_{index}" in panel


def test_feedback_marker_reprojects_as_elapsed_time_grows(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "column_a", 100)
        state = bar._bars["a"]  # noqa: SLF001
        state.rates = [0.0, 10.0, 20.0]
        state.latest_rate = 12.0
        bar._start_time = time.perf_counter() - 10.0  # noqa: SLF001

        bar.record_feedback_signal(event_kind="request_rate_limited", force=True)
        before_positions = _marker_positions(_last_panel_lines(tty_stream.getvalue()))
        assert before_positions

        bar._start_time = time.perf_counter() - 100.0  # noqa: SLF001
        bar.update("a", completed=20, success=20, force=True)
        after_positions = _marker_positions(_last_panel_lines(tty_stream.getvalue()))

        assert after_positions
        assert after_positions[0][1] < before_positions[0][1]


def test_control_sequences_are_removed_from_labels(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "col\x1b[31m_a\nsuffix", 100)
        bar.update("a", completed=10, success=10, force=True)

        clean = _clean(tty_stream.getvalue())
        assert "col_asuffix" in clean


def test_rate_samples_are_bounded() -> None:
    state = _BarState(label="col_a", total=1_000_000, start_time=0.0, last_sample_time=0.0)

    for index in range(_MAX_RATE_SAMPLES + 5):
        completed = (index + 1) * 10
        state.record_update(
            completed=completed,
            success=completed,
            failed=0,
            skipped=0,
            now=(index + 1) * _RATE_SAMPLE_INTERVAL_SECONDS,
        )

    assert len(state.rates) == _MAX_RATE_SAMPLES


def test_completed_rate_sample_is_not_diluted_by_later_updates() -> None:
    state = _BarState(label="col_a", total=100, start_time=0.0, last_sample_time=0.0)

    state.record_update(completed=100, success=100, failed=0, skipped=0, now=10.0)
    completed_rates = list(state.rates)
    completed_latest_rate = state.latest_rate

    for now in (12.0, 14.0, 16.0):
        state.record_update(completed=100, success=100, failed=0, skipped=0, now=now)

    assert state.rates == completed_rates
    assert state.latest_rate == completed_latest_rate


def test_sparse_rate_samples_span_chart_width() -> None:
    fitted = _fit_series([0.0, 10.0, 5.0], 7)

    assert len(fitted) == 7
    assert fitted[0] == 0.0
    assert fitted[3] == pytest.approx(10.0)
    assert fitted[-1] == 5.0


def test_frequent_updates_are_redraw_throttled(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.add_bar("b", "col_b", 100)
        bar.update_many({"a": (1, 1, 0, 0), "b": (2, 2, 0, 0)}, force=True)

        snapshot = tty_stream.getvalue()
        for i in range(50):
            bar.update_many({"a": (i, i, 0, 0), "b": (i * 2, i * 2, 0, 0)})

        assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 0

        bar.update("a", completed=50, success=50, force=True)
        assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 22
        assert bar.drawn_lines == 22


def test_log_interleaving_preserves_panel_height(tty_stream: FakeTTY) -> None:
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(tty_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    try:
        with TerminalThroughputPanel(stream=tty_stream) as bar:
            bar.add_bar("x", "col_x", 100)
            bar.add_bar("y", "col_y", 100)

            for i in range(10):
                bar.update("x", completed=i, success=i)
                root_logger.info("log at step %d", i)
                bar.update("y", completed=i, success=i)

            snapshot = tty_stream.getvalue()
            bar.update("x", completed=20, success=20, force=True)
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 22
    finally:
        root_logger.removeHandler(handler)


def test_narrow_terminal_keeps_panel_within_width(tty_stream: FakeTTY) -> None:
    narrow = os.terminal_size((36, 24))
    with patch.object(shutil, "get_terminal_size", return_value=narrow):
        with TerminalThroughputPanel(stream=tty_stream) as bar:
            bar.add_bar("a", "column 'verification_1'", 300)
            bar.update("a", completed=50, success=50, force=True)

            output = tty_stream.getvalue()
            for line in _last_panel_lines(output):
                assert len(line) <= 35


def test_update_many_single_redraw(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.add_bar("b", "col_b", 100)
        before = tty_stream.getvalue()

        bar.update_many({"a": (10, 10, 0, 0), "b": (20, 20, 0, 0)}, force=True)
        after = tty_stream.getvalue()

        new_output = after[len(before) :]
        assert new_output.count(CURSOR_UP_CLEAR) == 22

        clean = _clean(after)
        assert "10/100" in clean
        assert "20/100" in clean


def test_update_many_includes_failures_and_skips(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.update_many({"a": (10, 7, 2, 1), "unknown": (5, 5, 0, 0)}, force=True)

        clean = _clean(tty_stream.getvalue())
        assert "10/100" in clean
        assert "2 failed" in clean
        assert "1 skipped" in clean
        assert "unknown" not in clean


def test_remove_bar_redraws_panel(tty_stream: FakeTTY) -> None:
    with TerminalThroughputPanel(stream=tty_stream) as bar:
        bar.add_bar("a", "col_a", 100)
        bar.add_bar("b", "col_b", 100)

        snapshot = tty_stream.getvalue()
        bar.remove_bar("a")

        new_output = tty_stream.getvalue()[len(snapshot) :]
        assert new_output.count(CURSOR_UP_CLEAR) == 22
        panel = "\n".join(_last_panel_lines(tty_stream.getvalue()))
        assert "col_a" not in panel
        assert "col_b" in panel


def test_reporter_updates_and_logs_keep_drawn_lines_in_sync(tty_stream: FakeTTY) -> None:
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(tty_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    try:
        bar = TerminalThroughputPanel(stream=tty_stream)
        trackers = {
            "col_a": ProgressTracker(total_records=100, label="column 'a'", quiet=True),
            "col_b": ProgressTracker(total_records=100, label="column 'b'", quiet=True),
            "col_c": ProgressTracker(total_records=100, label="column 'c'", quiet=True),
        }

        with bar:
            reporter = AsyncProgressReporter(trackers, report_interval=0.1, progress_bar=bar)
            reporter.log_start(num_row_groups=1)
            panel = "\n".join(_last_panel_lines(tty_stream.getvalue()))
            assert "col_a" in panel
            assert "column 'a'" not in panel

            emit_token_usage_event(
                TokenUsageEvent(
                    model_alias="test",
                    model_name="test-model",
                    input_tokens=120,
                    output_tokens=30,
                )
            )
            assert bar._model_usage["test"].input_tokens == 120  # noqa: SLF001
            assert bar._model_usage["test"].output_tokens == 30  # noqa: SLF001

            snapshot = tty_stream.getvalue()
            reporter.record_success("col_a")
            assert tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR) == 0

            for i in range(49):
                if i % 10 == 0:
                    root_logger.info("Processing batch %d", i)
                reporter.record_success("col_b")
                reporter.record_skipped("col_c")

            snapshot = tty_stream.getvalue()
            reporter.log_final()
            assert bar.drawn_lines == 22
            clear_count = tty_stream.getvalue()[len(snapshot) :].count(CURSOR_UP_CLEAR)
            assert clear_count >= bar.drawn_lines
            assert clear_count % bar.drawn_lines == 0
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)


def test_reporter_records_feedback_markers_from_request_events(tty_stream: FakeTTY) -> None:
    trackers = {"col_a": ProgressTracker(total_records=100, label="column 'a'", quiet=True)}

    with TerminalThroughputPanel(stream=tty_stream) as bar:
        reporter = AsyncProgressReporter(trackers, report_interval=0.1, progress_bar=bar)
        try:
            emit_request_admission_event(
                RequestAdmissionEvent.capture("request_rate_limited", sequence=1),
            )
            assert len(bar._feedback_markers) == 1  # noqa: SLF001

            emit_request_admission_event(
                RequestAdmissionEvent.capture("request_wait_started", sequence=2),
            )
            assert len(bar._feedback_markers) == 1  # noqa: SLF001
        finally:
            reporter.close()

        emit_request_admission_event(
            RequestAdmissionEvent.capture("request_rate_limited", sequence=3),
        )
        assert len(bar._feedback_markers) == 1  # noqa: SLF001


def test_reporter_filters_global_events_by_run_id(tty_stream: FakeTTY) -> None:
    trackers = {"col_a": ProgressTracker(total_records=100, label="column 'a'", quiet=True)}

    with TerminalThroughputPanel(stream=tty_stream) as bar:
        reporter = AsyncProgressReporter(trackers, report_interval=0.1, progress_bar=bar, run_id="run-a")
        try:
            emit_token_usage_event(
                TokenUsageEvent(
                    model_alias="other",
                    model_name="other-model",
                    input_tokens=100,
                    output_tokens=10,
                    correlation=_correlation("run-b"),
                )
            )
            emit_token_usage_event(
                TokenUsageEvent(
                    model_alias="uncorrelated",
                    model_name="uncorrelated-model",
                    input_tokens=100,
                    output_tokens=10,
                )
            )
            assert not bar._model_usage  # noqa: SLF001

            emit_token_usage_event(
                TokenUsageEvent(
                    model_alias="owned",
                    model_name="owned-model",
                    input_tokens=120,
                    output_tokens=30,
                    correlation=_correlation("run-a"),
                )
            )
            assert set(bar._model_usage) == {"owned"}  # noqa: SLF001

            emit_request_admission_event(
                RequestAdmissionEvent.capture("request_rate_limited", sequence=1, correlation=_correlation("run-b"))
            )
            emit_request_admission_event(RequestAdmissionEvent.capture("request_rate_limited", sequence=2))
            assert not bar._feedback_markers  # noqa: SLF001

            emit_request_admission_event(
                RequestAdmissionEvent.capture("request_rate_limited", sequence=3, correlation=_correlation("run-a"))
            )
            assert len(bar._feedback_markers) == 1  # noqa: SLF001
        finally:
            reporter.close()
