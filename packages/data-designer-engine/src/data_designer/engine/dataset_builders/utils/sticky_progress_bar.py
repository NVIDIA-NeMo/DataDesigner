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
_DEFAULT_PANEL_HEIGHT = 16
_MIN_PANEL_HEIGHT = 9
_MIN_TERMINAL_WIDTH = 30
_MIN_REDRAW_INTERVAL_SECONDS = 0.75
_RATE_SAMPLE_INTERVAL_SECONDS = 2.0
_RATE_SMOOTHING_WINDOW = 3
_MAX_RATE_SAMPLES = 7200
_RATE_FORMAT = "{:6.1f} "
_Y_AXIS_RESERVED = 12


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

        should_sample = elapsed >= _RATE_SAMPLE_INTERVAL_SECONDS or bounded_completed >= self.total
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


class StickyProgressBar:
    """ANSI throughput chart panel that sticks to the bottom of the terminal.

    Log messages (via standard ``logging``) are rendered above the panel
    automatically. The panel redraws in-place after each update, tracks one
    records-per-second curve per active generation column, and keeps a bounded
    height so it does not take over the terminal.

    Usage::

        with StickyProgressBar() as bar:
            bar.add_bar("col_a", "column 'a'", total=100)
            for i in range(100):
                bar.update("col_a", completed=i + 1, success=i + 1)

    Falls back to a no-op on non-TTY streams (CI, pipes, notebooks).
    """

    def __init__(self, stream: TextIO | None = None, *, panel_height: int = _DEFAULT_PANEL_HEIGHT) -> None:
        self._stream = stream or sys.stderr
        self._is_tty = hasattr(self._stream, "isatty") and self._stream.isatty()
        self._bars: dict[str, _BarState] = {}
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

    def __enter__(self) -> StickyProgressBar:
        if self._is_tty and shutil.get_terminal_size().columns >= _MIN_TERMINAL_WIDTH:
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

        legend_capacity = 4 if panel_height >= 13 else max(1, panel_height - 9)
        chart_line_count = max(3, panel_height - 4 - legend_capacity)
        chart_height = chart_line_count - 1

        now = time.perf_counter()
        bars = list(self._bars.values())
        chart_lines = self._format_chart_lines(bars, inner_width, chart_height)
        legend_lines = self._format_legend_lines(bars, now, legend_capacity)

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

    def _format_chart_lines(self, bars: list[_BarState], inner_width: int, chart_height: int) -> list[str]:
        max_points = max(2, inner_width - _Y_AXIS_RESERVED)
        series = [_fit_series(_smooth_series(bar.rates), max_points) for bar in bars]
        max_rate = max((max(points) for points in series if points), default=0.0)
        chart = asciichartpy.plot(
            series,
            {
                "height": chart_height,
                "min": 0.0,
                "max": max(1.0, max_rate),
                "format": _RATE_FORMAT,
                "colors": [_CURVE_COLORS[i % len(_CURVE_COLORS)] for i in range(len(series))],
            },
        )
        lines = chart.splitlines()
        while len(lines) < chart_height + 1:
            lines.append("")
        return lines[: chart_height + 1]

    def _format_legend_lines(self, bars: list[_BarState], now: float, capacity: int) -> list[str]:
        lines: list[str] = []
        visible_bars = bars
        if len(bars) > capacity:
            visible_bars = bars[: max(0, capacity - 1)]

        for index, bar in enumerate(visible_bars):
            color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
            pct = (bar.completed / bar.total * 100) if bar.total > 0 else 100.0
            failed = f" | {_color(str(bar.failed) + ' failed', _FAILED)}" if bar.failed else ""
            skipped = f" | {bar.skipped} skipped" if bar.skipped else ""
            lines.append(
                f"{_color('●', color)} {bar.label}: {bar.completed}/{bar.total} "
                f"({pct:3.0f}%) | now {bar.latest_rate:5.1f} rec/s | "
                f"avg {bar.average_rate(now):5.1f}{failed}{skipped}"
            )

        if len(bars) > capacity:
            lines.append(f"{_MUTED}... {len(bars) - len(visible_bars)} more column(s){_RESET}")

        while len(lines) < capacity:
            lines.append("")
        return lines[:capacity]

    def _panel_line(self, text: str, inner_width: int) -> str:
        return f"{_BORDER}│{_RESET}{_fit_ansi(text, inner_width)}{_BORDER}│{_RESET}"

    def _border(self, left: str, fill: str, right: str, width: int) -> str:
        return f"{_BORDER}{left}{fill * (width - 2)}{right}{_RESET}"

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()
