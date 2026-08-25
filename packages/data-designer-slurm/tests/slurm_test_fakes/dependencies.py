# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

RequestT = TypeVar("RequestT")
ResolutionT = TypeVar("ResolutionT")
LockT = TypeVar("LockT")
InstallationT = TypeVar("InstallationT")


class FakeDependencyResolver(Generic[RequestT, ResolutionT]):
    """Return or raise exact scripted dependency-resolution outcomes."""

    def __init__(self, responses: Iterable[tuple[RequestT, ResolutionT | BaseException]]) -> None:
        self._responses = deque(responses)
        self.calls: list[RequestT] = []

    def resolve(self, request: RequestT) -> ResolutionT:
        """Resolve one expected request without consulting package indexes."""
        self.calls.append(request)
        if not self._responses:
            raise AssertionError("unexpected dependency resolution")
        expected, response = self._responses.popleft()
        if request != expected:
            raise AssertionError(f"expected dependency request {expected!r}, got {request!r}")
        if isinstance(response, BaseException):
            raise response
        return response

    def assert_complete(self) -> None:
        """Assert that every scripted resolution was consumed."""
        if self._responses:
            raise AssertionError(f"{len(self._responses)} dependency resolutions remain")


class FakeDependencyInstaller(Generic[LockT, InstallationT]):
    """Return or raise exact scripted dependency-install outcomes."""

    def __init__(
        self,
        responses: Iterable[tuple[tuple[LockT, Path], InstallationT | BaseException]],
    ) -> None:
        self._responses = deque(responses)
        self.calls: list[tuple[LockT, Path]] = []

    def install(self, lock: LockT, target: Path) -> InstallationT:
        """Install one expected lock without invoking an installer process."""
        call = (lock, target)
        self.calls.append(call)
        if not self._responses:
            raise AssertionError("unexpected dependency installation")
        expected, response = self._responses.popleft()
        if call != expected:
            raise AssertionError(f"expected dependency installation {expected!r}, got {call!r}")
        if isinstance(response, BaseException):
            raise response
        return response

    def assert_complete(self) -> None:
        """Assert that every scripted installation was consumed."""
        if self._responses:
            raise AssertionError(f"{len(self._responses)} dependency installations remain")
