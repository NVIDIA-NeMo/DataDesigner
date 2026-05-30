# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from data_designer.engine.observability import RuntimeCorrelation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsageEvent:
    model_alias: str
    model_name: str
    input_tokens: int
    output_tokens: int
    column: str | None = None
    correlation: RuntimeCorrelation | None = None


TokenUsageCallback = Callable[[TokenUsageEvent], None]

_callback_lock = Lock()
_callback_ids = itertools.count()
_callbacks: dict[int, TokenUsageCallback] = {}


def subscribe_token_usage(callback: TokenUsageCallback) -> Callable[[], None]:
    callback_id = next(_callback_ids)
    with _callback_lock:
        _callbacks[callback_id] = callback

    def unsubscribe() -> None:
        with _callback_lock:
            _callbacks.pop(callback_id, None)

    return unsubscribe


def emit_token_usage_event(event: TokenUsageEvent) -> None:
    with _callback_lock:
        callbacks = tuple(_callbacks.values())

    for callback in callbacks:
        try:
            callback(event)
        except Exception:
            logger.debug("Token usage event callback failed", exc_info=True)
