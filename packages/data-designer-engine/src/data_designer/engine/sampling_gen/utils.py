# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numbers
import os
import sys
import threading

import data_designer.lazy_heavy_imports as lazy


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    This function was taken from scikit-learn's utils module.
    Source GitHub: https://github.com/scikit-learn/scikit-learn

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from data_designer.engine.sampling_gen.utils import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is lazy.np.random:
        return lazy.np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return lazy.np.random.RandomState(seed)
    if isinstance(seed, lazy.np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)


_held_numpy_rng_lock: threading.RLock | None = None


def _numpy_global_rng_lock() -> threading.RLock | None:
    # Every draw from numpy's global RandomState, and seed(), holds its bit generator's RLock.
    # Only look it up when numpy.random is already imported, to keep the fork hooks from
    # importing it.
    numpy_random = sys.modules.get("numpy.random")
    if numpy_random is None:
        return None
    return numpy_random.mtrand._rand._bit_generator.lock


def _acquire_numpy_rng_lock_before_fork() -> None:
    # Hold the lock across fork so a child never inherits it locked by a thread that does not
    # exist there, which would block the child's reseed (and any later draw) forever. This waits
    # for a draw in progress on another thread to finish.
    global _held_numpy_rng_lock
    lock = _numpy_global_rng_lock()
    if lock is not None:
        lock.acquire()
    _held_numpy_rng_lock = lock


def _release_numpy_rng_lock_after_fork() -> None:
    global _held_numpy_rng_lock
    lock, _held_numpy_rng_lock = _held_numpy_rng_lock, None
    if lock is not None:
        lock.release()


def _reseed_shared_rngs_after_fork() -> None:
    # Samplers draw from numpy's global RandomState (check_random_state(None)) and Faker's
    # shared Random. CPython reseeds the stdlib random module in a forked child but not these,
    # so children forked after a run would all draw the same samples. Reseed them from OS
    # entropy, as CPython does for random; a child that needs reproducible draws seeds itself.
    # Only reseed modules already imported, to keep the hook from importing them.
    _release_numpy_rng_lock_after_fork()
    numpy_random = sys.modules.get("numpy.random")
    if numpy_random is not None:
        numpy_random.seed()
    faker_generator = sys.modules.get("faker.generator")
    if faker_generator is not None:
        faker_generator.random.seed()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_acquire_numpy_rng_lock_before_fork,
        after_in_parent=_release_numpy_rng_lock_after_fork,
        after_in_child=_reseed_shared_rngs_after_fork,
    )
