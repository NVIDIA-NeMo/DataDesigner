# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

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


def test_numpy_global_rng_lock_is_the_lock_draws_hold() -> None:
    # The fork hooks reach into numpy internals for this lock; fail here if numpy moves it.
    rng = check_random_state(None)  # imports numpy.random
    lock = _numpy_global_rng_lock()
    assert lock is not None
    finished = threading.Event()
    with lock:
        drawer = threading.Thread(target=lambda: (rng.random_sample(), finished.set()))
        drawer.start()
        assert not finished.wait(0.2), "a global RandomState draw did not wait for the lock"
    drawer.join(timeout=5)
    assert finished.is_set()


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
