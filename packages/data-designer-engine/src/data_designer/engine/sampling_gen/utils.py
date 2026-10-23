# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numbers
import os
import sys

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


def _reseed_shared_rngs_after_fork() -> None:
    # Samplers draw from numpy's global RandomState (check_random_state(None)) and Faker's
    # shared Random. CPython reseeds the stdlib random module in a forked child but not these,
    # so children forked after a run would all draw the same samples. Reseed them from OS
    # entropy, as CPython does for random; a child that needs reproducible draws seeds itself.
    # Only reseed modules already imported, to keep the hook from importing them.
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed()
    faker_generator = sys.modules.get("faker.generator")
    if faker_generator is not None:
        faker_generator.random.seed()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reseed_shared_rngs_after_fork)
