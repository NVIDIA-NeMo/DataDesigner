# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
LiteLLM overrides and customizations.

Note on imports: This module uses direct (eager) imports for litellm rather than lazy loading.
This is intentional because:

1. Class inheritance requires base classes to be resolved at class definition time,
   making lazy imports incompatible with our ThreadSafeCache and CustomRouter classes.

2. This module is already lazily loaded at the application level - it's only imported
   by facade.py, which itself is imported inside the create_model_registry() factory
   function. So litellm is only loaded when models are actually needed.

3. Attempting to use lazy imports here causes intermittent ImportErrors.
"""

from __future__ import annotations

import random
import threading

import httpx
import litellm
from litellm import RetryPolicy
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager
from litellm.router import Router
from pydantic import BaseModel, Field
from typing_extensions import override

from data_designer.logging import quiet_noisy_logger

DEFAULT_MAX_CALLBACKS = 1000


class LiteLLMRouterDefaultKwargs(BaseModel):
    ## Number of seconds to wait initially after a connection
    ## failure.
    initial_retry_after_s: float = 2.0

    ## Jitter percentage added during exponential backoff to
    ## smooth repeated retries over time.
    jitter_pct: float = 0.2

    ## Maximum number of seconds to wait for an API request
    ## before letting it die. Will trigger a retry.
    timeout: float = 60.0

    ## Sets the default retry policy, including the number
    ## of retries to use in particular scenarios.
    retry_policy: RetryPolicy = Field(
        default_factory=lambda: RetryPolicy(
            RateLimitErrorRetries=3,
            TimeoutErrorRetries=3,
        )
    )


class ThreadSafeCache(InMemoryCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._lock = threading.RLock()

    def get_cache(self, key, **kwargs):
        with self._lock:
            return super().get_cache(key, **kwargs)

    def set_cache(self, key, value, **kwargs):
        with self._lock:
            super().set_cache(key, value, **kwargs)

    def batch_get_cache(self, keys: list, **kwargs):
        with self._lock:
            return super().batch_get_cache(keys, **kwargs)

    def delete_cache(self, key):
        with self._lock:
            super().delete_cache(key)

    def evict_cache(self):
        with self._lock:
            super().evict_cache()

    def increment_cache(self, key, value: int, **kwargs) -> int:
        with self._lock:
            return super().increment_cache(key, value, **kwargs)

    def flush_cache(self):
        with self._lock:
            super().flush_cache()


class CustomRouter(Router):
    def __init__(
        self,
        *args,
        initial_retry_after_s: float,
        jitter_pct: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._initial_retry_after_s = initial_retry_after_s
        self._jitter_pct = jitter_pct

    def _extract_retry_delay_from_headers(self, e: Exception) -> int | float | None:
        """
        Most of this code logic was extracted directly from the parent
        `Router`'s `_time_to_sleep_before_retry` function. Our override
        of that method below should only affect requests where the server
        didn't explicitly return a desired retry-delay. If the server did
        return this info, we'll simply use that retry value returned here.
        """

        response_headers: httpx.Headers | None = None
        if hasattr(e, "response") and hasattr(e.response, "headers"):  # type: ignore
            response_headers = e.response.headers  # type: ignore
        if hasattr(e, "litellm_response_headers"):
            response_headers = e.litellm_response_headers  # type: ignore

        retry_after = litellm.utils._get_retry_after_from_exception_header(response_headers)

        # If the API asks us to wait a certain amount of time (and it's a reasonable amount), just do what it says.
        if retry_after is not None and 0 < retry_after <= 60:
            return retry_after
        else:
            return None

    @override
    def _time_to_sleep_before_retry(
        self,
        e: Exception,
        remaining_retries: int,
        num_retries: int,
        healthy_deployments: list | None = None,
        all_deployments: list | None = None,
    ) -> int | float:
        """
        Implements exponential backoff for retries.

        Technically, litellm's `Router` already implements some
        form of exponential backoff. However, that backoff
        is not customizable w.r.t jitter and initial delay
        timing. For that reason, we override this method to
        utilize our own custom instance variables, deferring
        to the existing implementation wherever we can.
        """

        # If the response headers indicated how long we should wait,
        # use that information.
        if retry_after := self._extract_retry_delay_from_headers(e):
            return retry_after

        return self.calculate_exponential_backoff(
            initial_retry_after_s=self._initial_retry_after_s,
            current_retry=num_retries - remaining_retries,
            jitter_pct=self._jitter_pct,
        )

    @staticmethod
    def calculate_exponential_backoff(initial_retry_after_s: float, current_retry: int, jitter_pct: float) -> float:
        sleep_s = initial_retry_after_s * (pow(2.0, current_retry))
        jitter = 1.0 + random.uniform(-jitter_pct, jitter_pct)
        return sleep_s * jitter


def apply_litellm_patches():
    litellm.in_memory_llm_clients_cache = ThreadSafeCache()

    # Workaround for the litellm issue described in https://github.com/BerriAI/litellm/issues/9792
    LoggingCallbackManager.MAX_CALLBACKS = DEFAULT_MAX_CALLBACKS

    quiet_noisy_logger("httpx")
    quiet_noisy_logger("LiteLLM")
    quiet_noisy_logger("LiteLLM Router")
