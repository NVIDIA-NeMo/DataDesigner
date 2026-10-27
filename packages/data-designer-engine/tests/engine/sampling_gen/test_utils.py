# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.sampling_gen.utils import _numpy_global_rng_lock, check_random_state
from tests.engine.fork_probe import requires_fork, run_fork_probe


@pytest.mark.parametrize(
    "test_case,input_value,expected_type,expected_seed",
    [
        ("none_input", None, "np.random.mtrand._rand", None),
        ("np_random_input", lazy.np.random, "np.random.mtrand._rand", None),
        ("integer_input", 42, "np.random.RandomState", 42),
        ("random_state_input", lazy.np.random.RandomState(123), "np.random.RandomState", 123),
    ],
)
def test_check_random_state_scenarios(test_case, input_value, expected_type, expected_seed):
    if test_case == "random_state_input":
        result = check_random_state(input_value)
        assert result is input_value
    else:
        result = check_random_state(input_value)

        if expected_type == "np.random.mtrand._rand":
            assert result is lazy.np.random.mtrand._rand
        elif expected_type == "np.random.RandomState":
            assert isinstance(result, lazy.np.random.RandomState)
            if expected_seed is not None:
                assert result.get_state()[1][0] == expected_seed


def test_check_random_state_invalid():
    with pytest.raises(ValueError, match="'invalid' cannot be used to seed a numpy.random.RandomState instance"):
        check_random_state("invalid")


@requires_fork
def test_forked_children_draw_different_samples() -> None:
    run_fork_probe(
        """
        import faker

        from data_designer.engine.sampling_gen.utils import check_random_state

        rng = check_random_state(None)
        fake = faker.Faker("en_US")
        rng.random_sample()  # a parent run has drawn from both, as sampler runs do
        fake.name()


        def draw(queue):
            queue.put((float(rng.random_sample()), [fake.name() for _ in range(5)]))


        first, second = run_forked_child(draw), run_forked_child(draw)
        assert first[0] != second[0], "forked children drew the same numpy samples"
        assert first[1] != second[1], "forked children drew the same Faker samples"
        """
    )


@requires_fork
def test_forked_child_reseed_does_not_change_parent_draws() -> None:
    run_fork_probe(
        """
        from data_designer.engine.sampling_gen.utils import check_random_state

        rng = check_random_state(None)
        rng.seed(7)
        expected = rng.random_sample(3).tolist()
        rng.seed(7)
        run_forked_child(lambda queue: queue.put(None))
        assert rng.random_sample(3).tolist() == expected, "forking reseeded the parent's RNG"
        """
    )


def _call_waits_for_lock(lock: threading.RLock, call: Callable[[], object]) -> bool:
    """Return whether ``call`` on another thread blocks while this thread holds ``lock``."""
    started, finished = threading.Event(), threading.Event()

    def run() -> None:
        started.set()
        call()
        finished.set()

    with lock:
        thread = threading.Thread(target=run)
        thread.start()
        started.wait()
        waited = not finished.wait(0.5)
    thread.join(timeout=5)
    assert finished.is_set(), "call did not finish after the lock was released"
    return waited


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: check_random_state(None).random_sample(), id="sampler-draw"),
        pytest.param(lambda: lazy.np.random.seed(), id="child-reseed"),
    ],
)
def test_fork_hook_lock_is_the_lock_global_random_state_takes(call: Callable[[], object]) -> None:
    # The fork hooks take numpy's lock through a private path (mtrand._rand._bit_generator.lock),
    # and the fork test holds the lock this helper returns, so it cannot tell whether numpy uses
    # it. Check against numpy itself: sampler draws and the child's reseed must wait for it.
    check_random_state(None)  # imports numpy.random
    lock = _numpy_global_rng_lock()
    assert lock is not None
    assert _call_waits_for_lock(lock, call), "numpy's global RandomState does not take the lock the fork hooks hold"


def test_call_waits_for_lock_detects_a_call_that_does_not_take_the_lock() -> None:
    # Control for the test above: a separate RandomState has its own lock and must not wait.
    check_random_state(None)
    lock = _numpy_global_rng_lock()
    assert lock is not None
    assert not _call_waits_for_lock(lock, lambda: check_random_state(1).random_sample())


@requires_fork
def test_fork_while_another_thread_draws_does_not_hang_child() -> None:
    run_fork_probe(
        """
        import threading
        import time

        from data_designer.engine.sampling_gen.utils import _numpy_global_rng_lock, check_random_state

        rng = check_random_state(None)
        rng.random_sample()
        lock = _numpy_global_rng_lock()
        held = threading.Event()


        def draw_in_progress():
            # Holds the lock as a draw on another thread does when fork is called.
            with lock:
                held.set()
                time.sleep(1.0)


        threading.Thread(target=draw_in_progress).start()
        held.wait()
        started = time.monotonic()
        value = run_forked_child(lambda queue: queue.put(float(rng.random_sample())))
        assert isinstance(value, float)
        assert time.monotonic() - started >= 0.9, "fork did not wait for the draw in progress"

        drawer = threading.Thread(target=rng.random_sample)
        drawer.start()
        drawer.join(timeout=5)
        assert not drawer.is_alive(), "the parent's RNG lock stayed held after fork"
        """
    )
