# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TextIO

BAR_FILLED = "█"
BAR_EMPTY = "░"


@dataclass
class _BarState:
    label: str
    total: int
    completed: int = 0
    success: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.perf_counter)


class StickyProgressBar:
    """ANSI progress bar that sticks to the bottom of the terminal.

    Log messages (via standard ``logging``) are rendered above the bar
    automatically. The bar redraws in-place after each update.

    Usage::

        with StickyProgressBar() as bar:
            bar.add_bar("col_a", "column 'a'", total=100)
            for i in range(100):
                bar.update("col_a", completed=i + 1, success=i + 1)
            bar.remove_bar("col_a")

    Falls back to a no-op on non-TTY streams (CI, pipes, notebooks).
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr
        self._is_tty = hasattr(self._stream, "isatty") and self._stream.isatty()
        self._bars: dict[str, _BarState] = {}
        self._lock = Lock()
        self._drawn_lines = 0
        self._active = False
        self._wrapped_handlers: list[tuple[logging.StreamHandler, object]] = []

    @property
    def is_active(self) -> bool:
        return self._active

    # -- context manager --

    def __enter__(self) -> StickyProgressBar:
        if self._is_tty:
            self._active = True
            self._wrap_handlers()
            self._write("\033[?25l")  # hide cursor
        return self

    def __exit__(self, *args: object) -> None:
        if self._active:
            with self._lock:
                self._clear_bars()
            self._write("\033[?25h")  # show cursor
            self._unwrap_handlers()
            self._active = False

    # -- public API --

    def add_bar(self, key: str, label: str, total: int) -> None:
        with self._lock:
            self._bars[key] = _BarState(label=label, total=total)
            if self._active:
                self._redraw()

    def update(
        self,
        key: str,
        *,
        completed: int,
        success: int = 0,
        failed: int = 0,
    ) -> None:
        with self._lock:
            if bar := self._bars.get(key):
                bar.completed = completed
                bar.success = success
                bar.failed = failed
                if self._active:
                    self._redraw()

    def remove_bar(self, key: str) -> None:
        with self._lock:
            self._bars.pop(key, None)
            if self._active:
                self._redraw()

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
                        self._redraw()

                return wrapped_emit

            handler.emit = _make_wrapper(original_emit)  # type: ignore[assignment]
            self._wrapped_handlers.append((handler, original_emit))

    def _unwrap_handlers(self) -> None:
        for handler, original_emit in self._wrapped_handlers:
            handler.emit = original_emit  # type: ignore[assignment]
        self._wrapped_handlers.clear()

    # -- drawing --

    def _clear_bars(self) -> None:
        """Clear drawn bar lines from the terminal. Caller must hold the lock."""
        if self._drawn_lines > 0:
            for _ in range(self._drawn_lines):
                self._write("\033[A\033[2K")
            self._write("\r\033[2K")
            self._drawn_lines = 0

    def _redraw(self) -> None:
        """Redraw all bars. Caller must hold the lock."""
        self._clear_bars()
        if not self._bars:
            return
        width = shutil.get_terminal_size().columns
        max_label = max(len(b.label) for b in self._bars.values())
        for bar in self._bars.values():
            line = self._format_bar(bar, width, max_label)
            self._write(line + "\n")
            self._drawn_lines += 1

    def _format_bar(self, bar: _BarState, width: int, label_width: int) -> str:
        pct = (bar.completed / bar.total * 100) if bar.total > 0 else 100.0
        elapsed = time.perf_counter() - bar.start_time
        rate = bar.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, bar.total - bar.completed)
        eta = f"{remaining / rate:.0f}s" if rate > 0 else "?"

        label = bar.label.ljust(label_width)
        total_w = len(str(bar.total))
        count_str = f"{bar.completed:>{total_w}}/{bar.total}"
        stats = f" {pct:3.0f}% | {count_str} | {rate:6.1f} rec/s | eta {eta:>4s}"
        if bar.failed > 0:
            stats += f" | {bar.failed} failed"

        bar_width = max(10, width - len(label) - len(stats) - 4)
        filled = int(bar_width * pct / 100)
        empty = bar_width - filled

        colored_bar = f"\033[32m{BAR_FILLED * filled}\033[90m{BAR_EMPTY * empty}\033[0m"
        return f"  {label} {colored_bar}{stats}"

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()
