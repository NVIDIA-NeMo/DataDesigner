# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized termination-signal coordination for the allocation runtime."""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType

_TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class TerminationSignalCoordinator:
    """Install, defer, and block allocation termination handling."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cleanup: Callable[[], None] | None = None
        self._defer_depth = 0
        self._deferred_signal: signal.Signals | None = None
        self._interrupted = False

    @contextmanager
    def interrupt_on_termination(self, cleanup: Callable[[], None]) -> Iterator[None]:
        """Install one cleanup-first termination handler for an allocation run."""
        if self._cleanup is not None:
            raise RuntimeError("termination signal coordination is already active")
        previous = {selected: signal.getsignal(selected) for selected in _TERMINATION_SIGNALS}
        self._cleanup = cleanup
        self._interrupted = False
        for selected in _TERMINATION_SIGNALS:
            signal.signal(selected, self._handle_termination)
        try:
            yield
        finally:
            for selected, handler in previous.items():
                signal.signal(selected, handler)
            self._cleanup = None
            self._deferred_signal = None
            self._defer_depth = 0

    @contextmanager
    def defer_termination(self) -> Iterator[None]:
        """Delay termination delivery until a child process is registered."""
        with self._lock:
            self._defer_depth += 1
        try:
            yield
        finally:
            deferred = self._finish_defer()
        if deferred is not None:
            os.kill(os.getpid(), deferred)

    @contextmanager
    def block_termination(self) -> Iterator[None]:
        """Block termination delivery while owned children are being stopped."""
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, _TERMINATION_SIGNALS)
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def _finish_defer(self) -> signal.Signals | None:
        with self._lock:
            self._defer_depth -= 1
            if self._defer_depth or self._interrupted:
                return None
            deferred = self._deferred_signal
            self._deferred_signal = None
            return deferred

    def _handle_termination(self, signum: int, frame: FrameType | None) -> None:
        del frame
        selected = signal.Signals(signum)
        with self._lock:
            if self._defer_depth:
                if self._deferred_signal is None:
                    self._deferred_signal = selected
                return
            if self._interrupted:
                return
            self._interrupted = True
            cleanup = self._cleanup
        try:
            if cleanup is not None:
                cleanup()
        finally:
            raise KeyboardInterrupt


__all__ = ["TerminationSignalCoordinator"]
