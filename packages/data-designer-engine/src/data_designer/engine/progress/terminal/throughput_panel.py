# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Sequence, TextIO

import asciichartpy

_ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_RESET = "\033[0m"
_BORDER = "\033[38;5;39m"
_TITLE = "\033[1;38;5;81m"
_MUTED = "\033[2;38;5;245m"
_FAILED = "\033[31m"
_OK = "\033[32m"
_TRACK = "\033[38;5;238m"
_FEEDBACK_MARKER = "\033[1;38;5;196m◆\033[0m"
_CURVE_COLORS = [
    asciichartpy.lightcyan,
    asciichartpy.lightgreen,
    asciichartpy.lightmagenta,
    asciichartpy.lightyellow,
    asciichartpy.lightblue,
    asciichartpy.lightred,
    asciichartpy.cyan,
    asciichartpy.green,
]
_DEFAULT_PANEL_HEIGHT = 22
_MIN_PANEL_HEIGHT = 9
_MIN_TERMINAL_WIDTH = 30
_MIN_TERMINAL_HEIGHT = _MIN_PANEL_HEIGHT + 1
_MIN_REDRAW_INTERVAL_SECONDS = 0.75
_RATE_SAMPLE_INTERVAL_SECONDS = 2.0
_RATE_SMOOTHING_WINDOW = 3
_MAX_RATE_SAMPLES = 7200
_MAX_FEEDBACK_MARKERS = 512
_RATE_FORMAT = "{:6.1f} "
_Y_AXIS_RESERVED = 12
_CHART_LINE_COUNT = 9
_MIN_CHART_LINE_COUNT = 3
_MIN_LEGEND_LABEL_WIDTH = 8
_MIN_MODEL_ALIAS_WIDTH = 10
_MIN_MODEL_NAME_WIDTH = 10
_RATE_COLUMN_WIDTH = 5
_INPUT_TOKEN_RATE_WIDTH = 8
_OUTPUT_TOKEN_RATE_WIDTH = 9
_LEGEND_COLUMN_GAP = 2
_MIN_PROGRESS_BAR_WIDTH = 6
_PROGRESS_BAR_CHAR = "━"
_NOW_RATE_HEADER = "now rec/s"
_AVG_RATE_HEADER = "avg rec/s"


_ProgressUpdate = tuple[int, int, int, int]


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _fit_ansi(text: str, width: int) -> str:
    visible = _visible_len(text)
    if visible > width:
        return _ANSI_RE.sub("", text)[:width]
    return text + (" " * (width - visible))


def _color(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _sanitize_label(label: str) -> str:
    return _CONTROL_RE.sub("", _ANSI_RE.sub("", label))


def _fit_plain(text: str, width: int) -> str:
    clean = _sanitize_label(text)
    return clean[:width].ljust(width)


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _smooth_series(series: Sequence[float], window: int = _RATE_SMOOTHING_WINDOW) -> list[float]:
    if window <= 1:
        return list(series)
    return [_average(series[max(0, i - window + 1) : i + 1]) for i in range(len(series))]


def _compress_series(series: Sequence[float], max_points: int) -> list[float]:
    if max_points <= 0:
        return []
    if len(series) <= max_points:
        return list(series) or [0.0]

    compressed: list[float] = []
    count = len(series)
    for bucket_index in range(max_points):
        start = int(bucket_index * count / max_points)
        end = int((bucket_index + 1) * count / max_points)
        bucket = series[start : max(end, start + 1)]
        compressed.append(_average(bucket))
    return compressed


def _expand_series(series: Sequence[float], point_count: int) -> list[float]:
    if point_count <= 0:
        return []
    if not series:
        return [0.0] * point_count
    if len(series) == 1:
        return [series[0]] * point_count

    expanded: list[float] = []
    source_last_index = len(series) - 1
    target_last_index = max(1, point_count - 1)
    for index in range(point_count):
        position = index * source_last_index / target_last_index
        left_index = int(position)
        right_index = min(left_index + 1, source_last_index)
        weight = position - left_index
        expanded.append(series[left_index] * (1 - weight) + series[right_index] * weight)
    return expanded


def _fit_series(series: Sequence[float], point_count: int) -> list[float]:
    if len(series) > point_count:
        return _compress_series(series, point_count)
    return _expand_series(series, point_count)


def _visible_index_of_any(text: str, chars: str) -> int | None:
    visible_index = 0
    index = 0
    while index < len(text):
        if match := _ANSI_RE.match(text, index):
            index = match.end()
            continue
        if text[index] in chars:
            return visible_index
        visible_index += 1
        index += 1
    return None


def _replace_visible_char(text: str, visible_index: int, replacement: str) -> str:
    output: list[str] = []
    current_visible_index = 0
    index = 0
    replaced = False
    while index < len(text):
        if match := _ANSI_RE.match(text, index):
            output.append(match.group())
            index = match.end()
            continue

        if current_visible_index == visible_index:
            output.append(replacement)
            replaced = True
        else:
            output.append(text[index])

        current_visible_index += 1
        index += 1

    if not replaced and visible_index >= current_visible_index:
        output.append(" " * (visible_index - current_visible_index))
        output.append(replacement)
    return "".join(output)


@dataclass
class _BarState:
    label: str
    total: int
    completed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    last_sample_time: float = field(default_factory=time.perf_counter)
    last_completed: int = 0
    latest_rate: float = 0.0
    rates: list[float] = field(default_factory=lambda: [0.0])

    def record_update(
        self,
        *,
        completed: int,
        success: int,
        failed: int,
        skipped: int,
        now: float,
    ) -> None:
        bounded_completed = min(max(completed, 0), self.total) if self.total > 0 else max(completed, 0)
        elapsed = now - self.last_sample_time
        self.completed = bounded_completed
        self.success = success
        self.failed = failed
        self.skipped = skipped

        already_complete = self.total > 0 and self.last_completed >= self.total
        should_sample = (
            elapsed >= _RATE_SAMPLE_INTERVAL_SECONDS or bounded_completed >= self.total
        ) and not already_complete
        if should_sample:
            delta_completed = max(0, bounded_completed - self.last_completed)
            sample_elapsed = max(elapsed, 0.001)
            rate = delta_completed / sample_elapsed
            self.rates.append(rate)
            if len(self.rates) > _MAX_RATE_SAMPLES:
                del self.rates[: len(self.rates) - _MAX_RATE_SAMPLES]
            self.latest_rate = _average(self.rates[-_RATE_SMOOTHING_WINDOW:])
            self.last_completed = bounded_completed
            self.last_sample_time = now

    def average_rate(self, now: float) -> float:
        elapsed = max(now - self.start_time, 0.001)
        return self.completed / elapsed if elapsed > 0 else 0.0


@dataclass
class _ModelUsageState:
    model_alias: str
    model_name: str
    start_time: float
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record_usage(self, *, model_name: str, input_tokens: int, output_tokens: int) -> None:
        self.model_name = model_name
        self.request_count += 1
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)

    def rpm(self, now: float) -> float:
        elapsed_minutes = max((now - self.start_time) / 60.0, 0.001)
        return self.request_count / elapsed_minutes

    def input_token_rate(self, now: float) -> float:
        elapsed = max(now - self.start_time, 0.001)
        return self.input_tokens / elapsed if elapsed > 0 else 0.0

    def output_token_rate(self, now: float) -> float:
        elapsed = max(now - self.start_time, 0.001)
        return self.output_tokens / elapsed if elapsed > 0 else 0.0


@dataclass(frozen=True)
class _FeedbackMarker:
    elapsed: float
    value: float
    event_kind: str


class TerminalThroughputPanel:
    """ANSI throughput chart panel that sticks to the bottom of the terminal.

    Log messages (via standard ``logging``) are rendered above the panel
    automatically. The panel redraws in-place after each update, tracks one
    records-per-second curve per active generation column. The chart keeps a
    stable height in normal terminals, but flexes down before the column and
    model tables overflow the viewport.

    Usage::

        with TerminalThroughputPanel() as bar:
            bar.add_bar("col_a", "column 'a'", total=100)
            for i in range(100):
                bar.update("col_a", completed=i + 1, success=i + 1)

    Falls back to a no-op on non-TTY streams (CI, pipes, notebooks).
    """

    def __init__(self, stream: TextIO | None = None, *, panel_height: int = _DEFAULT_PANEL_HEIGHT) -> None:
        self._stream = stream or sys.stderr
        self._is_tty = hasattr(self._stream, "isatty") and self._stream.isatty()
        self._bars: dict[str, _BarState] = {}
        self._model_usage: dict[str, _ModelUsageState] = {}
        self._feedback_markers: list[_FeedbackMarker] = []
        self._lock = Lock()
        self._drawn_lines = 0
        self._active = False
        self._wrapped_handlers: list[tuple[logging.StreamHandler, object]] = []
        self._panel_height = max(_MIN_PANEL_HEIGHT, panel_height)
        self._start_time = time.perf_counter()
        self._last_redraw_time: float = 0.0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def drawn_lines(self) -> int:
        return self._drawn_lines

    # -- context manager --

    def __enter__(self) -> TerminalThroughputPanel:
        terminal_size = shutil.get_terminal_size()
        if (
            self._is_tty
            and terminal_size.columns >= _MIN_TERMINAL_WIDTH
            and terminal_size.lines >= _MIN_TERMINAL_HEIGHT
        ):
            self._active = True
            self._start_time = time.perf_counter()
            self._last_redraw_time = 0.0
            self._wrap_handlers()
            self._write("\033[?25l")  # hide cursor
        return self

    def __exit__(self, *args: object) -> None:
        if self._active:
            self._write("\033[?25h")  # show cursor
            self._unwrap_handlers()
            self._active = False
            self._drawn_lines = 0

    # -- public API --

    def add_bar(self, key: str, label: str, total: int) -> None:
        with self._lock:
            self._bars[key] = _BarState(label=_sanitize_label(label), total=total)
            if self._active:
                self._redraw(force=True)

    def update(
        self,
        key: str,
        *,
        completed: int,
        success: int = 0,
        failed: int = 0,
        skipped: int = 0,
        force: bool = False,
    ) -> None:
        with self._lock:
            if bar := self._bars.get(key):
                now = time.perf_counter()
                bar.record_update(
                    completed=completed,
                    success=success,
                    failed=failed,
                    skipped=skipped,
                    now=now,
                )
                if self._active:
                    self._redraw_if_due(now, force=force)

    def update_many(self, updates: dict[str, _ProgressUpdate], *, force: bool = False) -> None:
        with self._lock:
            now = time.perf_counter()
            for key, update in updates.items():
                if bar := self._bars.get(key):
                    completed, success, failed, skipped = update
                    bar.record_update(
                        completed=completed,
                        success=success,
                        failed=failed,
                        skipped=skipped,
                        now=now,
                    )
            if self._active:
                self._redraw_if_due(now, force=force)

    def record_model_usage(
        self,
        *,
        model_alias: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        force: bool = False,
    ) -> None:
        with self._lock:
            now = time.perf_counter()
            alias = _sanitize_label(model_alias) or "(unknown)"
            name = _sanitize_label(model_name) or "(unknown)"
            if state := self._model_usage.get(alias):
                state.record_usage(model_name=name, input_tokens=input_tokens, output_tokens=output_tokens)
            else:
                self._model_usage[alias] = _ModelUsageState(
                    model_alias=alias,
                    model_name=name,
                    start_time=self._start_time,
                    request_count=1,
                    input_tokens=max(0, input_tokens),
                    output_tokens=max(0, output_tokens),
                )
            if self._active:
                self._redraw_if_due(now, force=force)

    def record_feedback_signal(self, *, event_kind: str, force: bool = False) -> None:
        with self._lock:
            now = time.perf_counter()
            self._feedback_markers.append(
                _FeedbackMarker(
                    elapsed=max(now - self._start_time, 0.0),
                    value=max((bar.latest_rate for bar in self._bars.values()), default=0.0),
                    event_kind=_sanitize_label(event_kind),
                )
            )
            if len(self._feedback_markers) > _MAX_FEEDBACK_MARKERS:
                del self._feedback_markers[: len(self._feedback_markers) - _MAX_FEEDBACK_MARKERS]
            if self._active:
                self._redraw_if_due(now, force=force)

    def remove_bar(self, key: str) -> None:
        with self._lock:
            self._bars.pop(key, None)
            if self._active:
                self._redraw(force=True)

    # -- handler wrapping --

    def _wrap_handlers(self) -> None:
        """Wrap stderr logging handlers so log lines render above the bars."""
        root = logging.getLogger()
        for handler in root.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            if getattr(handler, "stream", None) is not self._stream:
                continue
            original_emit = handler.emit

            def _make_wrapper(orig: object) -> object:
                def wrapped_emit(record: logging.LogRecord) -> None:
                    with self._lock:
                        self._clear_bars()
                        orig(record)  # type: ignore[operator]
                        self._redraw(force=True)

                return wrapped_emit

            handler.emit = _make_wrapper(original_emit)  # type: ignore[assignment]
            self._wrapped_handlers.append((handler, original_emit))

    def _unwrap_handlers(self) -> None:
        for handler, original_emit in self._wrapped_handlers:
            handler.emit = original_emit  # type: ignore[assignment]
        self._wrapped_handlers.clear()

    # -- drawing --

    def _clear_bars(self) -> None:
        """Clear drawn panel lines from the terminal. Caller must hold the lock."""
        if self._drawn_lines > 0:
            for _ in range(self._drawn_lines):
                self._write("\033[A\033[2K")
            self._write("\r\033[2K")
            self._drawn_lines = 0

    def _redraw_if_due(self, now: float, *, force: bool = False) -> None:
        if force or self._drawn_lines == 0 or now - self._last_redraw_time >= _MIN_REDRAW_INTERVAL_SECONDS:
            self._redraw(force=True, now=now)

    def _redraw(self, *, force: bool = False, now: float | None = None) -> None:
        """Redraw the chart panel. Caller must hold the lock."""
        if not force:
            current_time = time.perf_counter() if now is None else now
            if self._drawn_lines > 0 and current_time - self._last_redraw_time < _MIN_REDRAW_INTERVAL_SECONDS:
                return
        self._clear_bars()
        if not self._bars:
            return
        lines = self._format_panel()
        for line in lines:
            self._write(line + "\n")
        self._drawn_lines = len(lines)
        self._last_redraw_time = time.perf_counter() if now is None else now

    def _format_panel(self) -> list[str]:
        terminal_size = shutil.get_terminal_size()
        panel_width = max(4, terminal_size.columns - 1)
        panel_height = min(self._panel_height, max(_MIN_PANEL_HEIGHT, terminal_size.lines - 1))
        inner_width = panel_width - 2

        now = time.perf_counter()
        bars = list(self._bars.values())
        model_usage = list(self._model_usage.values())
        legend_lines = self._format_legend_lines(bars, model_usage, now, inner_width)

        body_capacity = max(1, panel_height - 4)
        available_chart_lines = body_capacity - len(legend_lines)
        if available_chart_lines >= _CHART_LINE_COUNT:
            chart_line_count = _CHART_LINE_COUNT
        else:
            chart_line_count = max(_MIN_CHART_LINE_COUNT, available_chart_lines)
        minimum_legend_capacity = max(0, body_capacity - chart_line_count)
        while len(legend_lines) < minimum_legend_capacity:
            legend_lines.append("")

        chart_height = chart_line_count - 1
        chart_lines = self._format_chart_lines(bars, inner_width, chart_height, now)

        lines = [
            self._border("╭", "─", "╮", panel_width),
            self._panel_line(self._format_header(bars, now), inner_width),
        ]
        lines.extend(self._panel_line(line, inner_width) for line in chart_lines)
        lines.append(self._border("├", "─", "┤", panel_width))
        lines.extend(self._panel_line(line, inner_width) for line in legend_lines)
        lines.append(self._border("╰", "─", "╯", panel_width))
        return lines

    def _format_header(self, bars: list[_BarState], now: float) -> str:
        elapsed = max(now - self._start_time, 0.0)
        completed = sum(bar.completed for bar in bars)
        total = sum(bar.total for bar in bars)
        latest_rate = sum(bar.latest_rate for bar in bars)
        failed = sum(bar.failed for bar in bars)
        skipped = sum(bar.skipped for bar in bars)
        failed_text = _color(f"{failed} failed", _FAILED) if failed else _color("0 failed", _OK)
        skipped_text = f" | {skipped} skipped" if skipped else ""
        return (
            f"{_TITLE}Throughput{_RESET} "
            f"{_MUTED}rec/s | {elapsed:5.1f}s | {completed}/{total} | "
            f"now {latest_rate:6.1f}{skipped_text} | {_RESET}{failed_text}"
        )

    def _format_chart_lines(
        self,
        bars: list[_BarState],
        inner_width: int,
        chart_height: int,
        now: float,
    ) -> list[str]:
        max_points = max(2, inner_width - _Y_AXIS_RESERVED)
        series = [_fit_series(_smooth_series(bar.rates), max_points) for bar in bars]
        max_rate = max((max(points) for points in series if points), default=0.0)
        chart_max = max(1.0, max_rate)
        chart = asciichartpy.plot(
            series,
            {
                "height": chart_height,
                "min": 0.0,
                "max": chart_max,
                "format": _RATE_FORMAT,
                "colors": [_CURVE_COLORS[i % len(_CURVE_COLORS)] for i in range(len(series))],
            },
        )
        lines = chart.splitlines()
        while len(lines) < chart_height + 1:
            lines.append("")
        return self._overlay_feedback_markers(
            lines[: chart_height + 1],
            current_elapsed=max(now - self._start_time, 0.001),
            chart_max=chart_max,
            point_count=max_points,
            chart_height=chart_height,
        )

    def _overlay_feedback_markers(
        self,
        lines: list[str],
        *,
        current_elapsed: float,
        chart_max: float,
        point_count: int,
        chart_height: int,
    ) -> list[str]:
        if not self._feedback_markers or not lines:
            return lines

        marked_lines = list(lines)
        plot_column_count = max(1, point_count - 1)
        for marker in self._feedback_markers:
            marker_elapsed = min(max(marker.elapsed, 0.0), current_elapsed)
            x_index = int(round(marker_elapsed / current_elapsed * (plot_column_count - 1)))
            y_value = min(max(marker.value, 0.0), chart_max)
            row_index = int(round((chart_max - y_value) / chart_max * chart_height))
            row_index = min(max(row_index, 0), len(marked_lines) - 1)
            axis_index = _visible_index_of_any(marked_lines[row_index], "┼┤")
            if axis_index is None:
                continue
            marked_lines[row_index] = _replace_visible_char(
                marked_lines[row_index],
                axis_index + 1 + x_index,
                _FEEDBACK_MARKER,
            )
        return marked_lines

    def _format_legend_lines(
        self,
        bars: list[_BarState],
        model_usage: list[_ModelUsageState],
        now: float,
        inner_width: int,
    ) -> list[str]:
        lines = self._format_column_table_lines(bars, now, inner_width)
        if model_usage:
            lines.append("")
            lines.extend(self._format_model_table_lines(model_usage, now, inner_width))
        return lines

    def _format_column_table_lines(
        self,
        bars: list[_BarState],
        now: float,
        inner_width: int,
    ) -> list[str]:
        lines: list[str] = []

        include_status = any(bar.failed or bar.skipped for bar in bars)
        label_width, done_width, rate_width, status_width, progress_width = self._column_table_widths(
            bars,
            now,
            include_status=include_status,
            inner_width=inner_width,
        )
        lines.append(
            self._format_legend_table_line(
                marker="",
                label="column",
                done="done",
                now_value=_NOW_RATE_HEADER,
                avg_value=_AVG_RATE_HEADER,
                status="status" if include_status else None,
                label_width=label_width,
                done_width=done_width,
                rate_width=rate_width,
                status_width=status_width,
                progress_bar="",
                progress_width=progress_width,
            )
        )

        for index, bar in enumerate(bars):
            color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
            lines.append(
                self._format_legend_table_line(
                    marker=_color("●", color),
                    label=bar.label,
                    done=self._format_done(bar),
                    now_value=f"{bar.latest_rate:.1f}",
                    avg_value=f"{bar.average_rate(now):.1f}",
                    status=self._format_status(bar) if include_status else None,
                    label_width=label_width,
                    done_width=done_width,
                    rate_width=rate_width,
                    status_width=status_width,
                    progress_bar=self._format_progress_bar(bar, progress_width, color),
                    progress_width=progress_width,
                )
            )

        return lines

    def _column_table_widths(
        self,
        bars: list[_BarState],
        now: float,
        *,
        include_status: bool,
        inner_width: int,
    ) -> tuple[int, int, int, int, int]:
        done_width = max(len("done"), *(len(self._format_done(bar)) for bar in bars))
        rate_width = max(
            len(_NOW_RATE_HEADER),
            len(_AVG_RATE_HEADER),
            _RATE_COLUMN_WIDTH,
            *(len(f"{value:.1f}") for bar in bars for value in (bar.latest_rate, bar.average_rate(now))),
        )
        status_width = 0
        if include_status:
            status_width = max(len("status"), *(len(self._format_status(bar)) for bar in bars))

        separator_count = 4 + int(include_status)
        fixed_width_without_label_or_progress = (
            2 + (separator_count * _LEGEND_COLUMN_GAP) + done_width + (rate_width * 2) + status_width
        )
        content_label_width = max(len("column"), *(len(_sanitize_label(bar.label)) for bar in bars))
        desired_label_width = max(_MIN_LEGEND_LABEL_WIDTH, content_label_width)
        available_width = inner_width - fixed_width_without_label_or_progress

        if available_width >= desired_label_width + _MIN_PROGRESS_BAR_WIDTH:
            label_width = desired_label_width
            progress_width = available_width - label_width
        elif available_width >= _MIN_LEGEND_LABEL_WIDTH + _MIN_PROGRESS_BAR_WIDTH:
            progress_width = _MIN_PROGRESS_BAR_WIDTH
            label_width = available_width - progress_width
        else:
            label_width = max(_MIN_LEGEND_LABEL_WIDTH, min(desired_label_width, max(0, available_width)))
            progress_width = max(0, available_width - label_width)

        return label_width, done_width, rate_width, status_width, progress_width

    def _format_model_table_lines(
        self,
        model_usage: list[_ModelUsageState],
        now: float,
        inner_width: int,
    ) -> list[str]:
        lines: list[str] = []

        alias_width, model_width, rpm_width, input_width, output_width = self._model_table_widths(
            model_usage,
            now,
            inner_width,
        )
        lines.append(
            self._format_model_table_line(
                model_alias="model alias",
                model_name="model name",
                rpm="rpm",
                input_token_rate="in tok/s",
                output_token_rate="out tok/s",
                alias_width=alias_width,
                model_width=model_width,
                rpm_width=rpm_width,
                input_width=input_width,
                output_width=output_width,
                header=True,
            )
        )

        for state in model_usage:
            lines.append(
                self._format_model_table_line(
                    model_alias=state.model_alias,
                    model_name=state.model_name,
                    rpm=f"{state.rpm(now):.1f}",
                    input_token_rate=f"{state.input_token_rate(now):.1f}",
                    output_token_rate=f"{state.output_token_rate(now):.1f}",
                    alias_width=alias_width,
                    model_width=model_width,
                    rpm_width=rpm_width,
                    input_width=input_width,
                    output_width=output_width,
                    header=False,
                )
            )

        return lines

    def _model_table_widths(
        self,
        model_usage: list[_ModelUsageState],
        now: float,
        inner_width: int,
    ) -> tuple[int, int, int, int, int]:
        rpm_width = max(
            len("rpm"),
            _RATE_COLUMN_WIDTH,
            *(len(f"{state.rpm(now):.1f}") for state in model_usage),
        )
        input_width = max(
            len("in tok/s"),
            _INPUT_TOKEN_RATE_WIDTH,
            *(len(f"{state.input_token_rate(now):.1f}") for state in model_usage),
        )
        output_width = max(
            len("out tok/s"),
            _OUTPUT_TOKEN_RATE_WIDTH,
            *(len(f"{state.output_token_rate(now):.1f}") for state in model_usage),
        )

        fixed_width_without_text = 2 + (4 * _LEGEND_COLUMN_GAP) + rpm_width + input_width + output_width
        available_text_width = inner_width - fixed_width_without_text
        desired_alias_width = max(
            _MIN_MODEL_ALIAS_WIDTH,
            len("model alias"),
            *(len(state.model_alias) for state in model_usage),
        )
        desired_model_width = max(
            _MIN_MODEL_NAME_WIDTH,
            len("model name"),
            *(len(state.model_name) for state in model_usage),
        )

        if available_text_width >= desired_alias_width + desired_model_width:
            alias_width = desired_alias_width
            model_width = desired_model_width
        elif available_text_width >= _MIN_MODEL_ALIAS_WIDTH + _MIN_MODEL_NAME_WIDTH:
            alias_width = min(desired_alias_width, max(_MIN_MODEL_ALIAS_WIDTH, available_text_width // 2))
            model_width = available_text_width - alias_width
        else:
            alias_width = _MIN_MODEL_ALIAS_WIDTH
            model_width = _MIN_MODEL_NAME_WIDTH

        return alias_width, model_width, rpm_width, input_width, output_width

    def _format_progress_bar(self, bar: _BarState, width: int, color: str) -> str:
        if width <= 0:
            return ""

        if bar.total <= 0:
            fraction = 1.0
        else:
            fraction = min(max(bar.completed / bar.total, 0.0), 1.0)

        filled_width = min(width, int(round(width * fraction)))
        if bar.completed > 0:
            filled_width = max(1, filled_width)
        empty_width = width - filled_width
        return f"{color}{_PROGRESS_BAR_CHAR * filled_width}{_TRACK}{_PROGRESS_BAR_CHAR * empty_width}{_RESET}"

    def _format_legend_table_line(
        self,
        *,
        marker: str,
        label: str,
        done: str,
        now_value: str,
        avg_value: str,
        status: str | None,
        label_width: int,
        done_width: int,
        rate_width: int,
        status_width: int,
        progress_bar: str,
        progress_width: int,
    ) -> str:
        marker_text = f"{marker} " if marker else "  "
        gap = " " * _LEGEND_COLUMN_GAP
        line = (
            f"{marker_text}{_fit_plain(label, label_width)}{gap}{now_value:>{rate_width}}{gap}{avg_value:>{rate_width}}"
        )
        if status is not None:
            line = f"{line}{gap}{status:>{status_width}}"
        line = f"{line}{gap}{done:>{done_width}}"
        if progress_width > 0:
            line = f"{line}{gap}{_fit_ansi(progress_bar, progress_width)}"
        if marker:
            return line
        return f"{_MUTED}{line}{_RESET}"

    def _format_model_table_line(
        self,
        *,
        model_alias: str,
        model_name: str,
        rpm: str,
        input_token_rate: str,
        output_token_rate: str,
        alias_width: int,
        model_width: int,
        rpm_width: int,
        input_width: int,
        output_width: int,
        header: bool,
    ) -> str:
        gap = " " * _LEGEND_COLUMN_GAP
        line = (
            f"  {_fit_plain(model_alias, alias_width)}{gap}{_fit_plain(model_name, model_width)}"
            f"{gap}{rpm:>{rpm_width}}{gap}{input_token_rate:>{input_width}}"
            f"{gap}{output_token_rate:>{output_width}}"
        )
        if header:
            return f"{_MUTED}{line}{_RESET}"
        return line

    def _format_done(self, bar: _BarState) -> str:
        pct = (bar.completed / bar.total * 100) if bar.total > 0 else 100.0
        return f"{bar.completed}/{bar.total} {pct:3.0f}%"

    def _format_status(self, bar: _BarState) -> str:
        parts: list[str] = []
        if bar.failed:
            parts.append(f"{bar.failed} failed")
        if bar.skipped:
            parts.append(f"{bar.skipped} skipped")
        return ", ".join(parts) if parts else "ok"

    def _panel_line(self, text: str, inner_width: int) -> str:
        return f"{_BORDER}│{_RESET}{_fit_ansi(text, inner_width)}{_BORDER}│{_RESET}"

    def _border(self, left: str, fill: str, right: str, width: int) -> str:
        return f"{_BORDER}{left}{fill * (width - 2)}{right}{_RESET}"

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()
