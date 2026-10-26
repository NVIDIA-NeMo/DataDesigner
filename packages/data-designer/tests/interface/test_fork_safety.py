# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys

import pytest

# Runs in a fresh interpreter so the forked parent is small and holds only what the
# scenario creates, not whatever earlier tests left in the pytest worker.
_PREVIEW_IN_FORKED_CHILD = """
import multiprocessing
import sys

import data_designer.config as dd
from data_designer.interface import DataDesigner


def preview():
    builder = dd.DataDesignerConfigBuilder()
    builder.add_column(
        dd.SamplerColumnConfig(
            name="category",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b", "c"]),
        )
    )
    assert len(DataDesigner().preview(builder, num_records=3).dataset) == 3


preview()  # the parent starts DataDesigner's background event loop
child = multiprocessing.get_context("fork").Process(target=preview)
child.start()
child.join(timeout=30)
if child.is_alive():
    child.kill()
    child.join()
    sys.exit("forked child hung in preview()")
sys.exit(child.exitcode)
"""


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires the fork start method")
def test_preview_in_child_forked_after_parent_preview() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PREVIEW_IN_FORKED_CHILD],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "NEMO_TELEMETRY_ENABLED": "false"},
    )
    assert result.returncode == 0, f"preview() in a forked child failed:\n{result.stderr[-4000:]}"


_PREVIEW_IN_SIBLING_FORKED_CHILDREN = """
import json
import multiprocessing
import sys

import data_designer.config as dd
from data_designer.interface import DataDesigner

builder = dd.DataDesignerConfigBuilder()
builder.add_column(
    dd.SamplerColumnConfig(
        name="category",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=list("abcdefgh")),
    )
)
builder.add_column(
    dd.SamplerColumnConfig(
        name="uniform",
        sampler_type=dd.SamplerType.UNIFORM,
        params=dd.UniformSamplerParams(low=0, high=1000),
    )
)


def preview(queue=None):
    dataset = DataDesigner().preview(builder, num_records=6).dataset
    if queue is not None:
        queue.put(json.dumps(dataset.to_dict(orient="list")))


def preview_in_child():
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=preview, args=(queue,))
    child.start()
    child.join(timeout=30)
    if child.is_alive():
        child.kill()
        child.join()
        sys.exit("forked child hung in preview()")
    if child.exitcode != 0:
        sys.exit(f"forked child exited with code {child.exitcode}")
    return queue.get(timeout=5)


preview()  # the parent's run advances the shared sampler RNG
if preview_in_child() == preview_in_child():
    sys.exit("sibling forked children generated identical sampler data")
"""


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires the fork start method")
def test_sibling_forked_children_generate_different_sampler_data() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PREVIEW_IN_SIBLING_FORKED_CHILDREN],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "NEMO_TELEMETRY_ENABLED": "false"},
    )
    assert result.returncode == 0, f"sibling forked children failed:\n{result.stderr[-4000:]}"
