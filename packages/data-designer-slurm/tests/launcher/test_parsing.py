# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from data_designer.slurm.launcher import QueueRecord, SlurmParseError
from data_designer.slurm.launcher.parsing import (
    parse_accounting,
    parse_gpu_counts,
    parse_queue,
    parse_state,
    parse_submission,
)
from data_designer.slurm.state import SchedulerIdentity, SchedulerState

GOLDEN_DIRECTORY = Path(__file__).parents[1] / "slurm_test_fakes" / "golden" / "slurm"
OVERSIZED_DECIMAL = "9" * 5000


@pytest.mark.parametrize(
    ("output", "expected_job_id", "expected_cluster"),
    (("4101\n", 4101, None), ("4101;primary\n", 4101, "primary")),
)
def test_parse_submission_accepts_parsable_sbatch_output(
    output: str,
    expected_job_id: int,
    expected_cluster: str | None,
) -> None:
    submission = parse_submission(output)

    assert submission.array_job_id == expected_job_id
    assert submission.cluster_name == expected_cluster


@pytest.mark.parametrize("output", ("", "0", "Submitted batch job 4101", "٤١٠١", "4101;", "4101;bad name"))
def test_parse_submission_rejects_malformed_output(output: str) -> None:
    with pytest.raises(SlurmParseError, match="invalid"):
        parse_submission(output)


def test_parse_queue_normalizes_active_array_tasks() -> None:
    records = parse_queue((GOLDEN_DIRECTORY / "squeue_active.txt").read_text())

    assert records == (
        _make_queue_record(0, SchedulerState.PENDING),
        _make_queue_record(1, SchedulerState.RUNNING),
    )


@pytest.mark.parametrize(
    ("raw_state", "expected"),
    (
        ("CONFIGURING", SchedulerState.PENDING),
        ("COMPLETING", SchedulerState.RUNNING),
        ("COMPLETED+", SchedulerState.COMPLETED),
        ("CANCELLED by 1234", SchedulerState.CANCELLED),
        ("TIMEOUT", SchedulerState.TIMED_OUT),
        ("NODE_FAIL", SchedulerState.NODE_FAILED),
        ("PREEMPTED", SchedulerState.PREEMPTED),
        ("REQUEUED", SchedulerState.REQUEUED),
        ("OUT_OF_MEMORY", SchedulerState.OUT_OF_MEMORY),
        ("A_NEW_STATE", SchedulerState.UNKNOWN),
    ),
)
def test_parse_state_normalizes_long_slurm_spellings(raw_state: str, expected: SchedulerState) -> None:
    assert parse_state(raw_state) is expected


def test_parse_accounting_normalizes_terminal_rows_and_ignores_step_rows() -> None:
    output = (
        "4101|RUNNING|0:0\n" + (GOLDEN_DIRECTORY / "sacct_retry_terminal.txt").read_text() + "4101_0.batch|FAILED|1:0\n"
    )

    records = parse_accounting(output)

    assert tuple(record.state for record in records) == (
        SchedulerState.TIMED_OUT,
        SchedulerState.NODE_FAILED,
        SchedulerState.PREEMPTED,
        SchedulerState.REQUEUED,
        SchedulerState.OUT_OF_MEMORY,
        SchedulerState.CANCELLED,
    )
    assert records[0].exit_code.status == 0
    assert records[0].exit_code.signal == 125


def test_empty_scheduler_output_preserves_absent_evidence_for_reconciliation() -> None:
    assert parse_queue("") == ()
    assert parse_accounting("\n") == ()


@pytest.mark.parametrize(
    ("parser", "output", "message"),
    (
        (parse_queue, "malformed scheduler output\n", "two fields"),
        (parse_queue, "4101|RUNNING\n", "array-task ID"),
        (parse_queue, "4101_0|RUNNING\n4101_0|PENDING\n", "duplicates"),
        (parse_accounting, "4101_0|FAILED\n", "three fields"),
        (parse_accounting, "4101_0|FAILED|not-an-exit-code\n", "exit code"),
        (parse_accounting, "garbage.step|FAILED|1:0\n", "array-task ID"),
        (parse_queue, "4101_0|COMPLETED unexpectedly\n", "unexpected whitespace"),
    ),
)
def test_scheduler_parsers_reject_malformed_or_ambiguous_rows(
    parser: Callable[[str], object],
    output: str,
    message: str,
) -> None:
    with pytest.raises(SlurmParseError, match=message):
        parser(output)


@pytest.mark.parametrize(
    ("parser", "output", "message"),
    (
        (parse_submission, OVERSIZED_DECIMAL, "invalid job ID"),
        (parse_queue, f"4101_{OVERSIZED_DECIMAL}|RUNNING", "array-task ID"),
        (parse_accounting, f"4101_0|FAILED|{OVERSIZED_DECIMAL}:0", "exit code"),
        (parse_gpu_counts, f"gpu:{OVERSIZED_DECIMAL}", "invalid GPU resource"),
    ),
    ids=("submission-job-id", "queue-task-id", "accounting-exit-code", "gpu-count"),
)
def test_parsers_normalize_oversized_numeric_fields(
    parser: Callable[[str], object],
    output: str,
    message: str,
) -> None:
    with pytest.raises(SlurmParseError, match=message):
        parser(output)


@pytest.mark.parametrize(
    ("output", "expected"),
    (
        ("gpu:2\n", (2,)),
        ("gpu:a100:8(S:0-7)\n", (8,)),
        ("gpu:a100:4(S:0-1,4-5)\n", (4,)),
        ("mps:100,gpu:a100:4\n", (4,)),
        ("gpu:a100:4,gpu:h100:4\n(null)\n", (8,)),
        ("(null)\nN/A\n", ()),
    ),
)
def test_parse_gpu_counts_normalizes_configured_gres(output: str, expected: tuple[int, ...]) -> None:
    assert parse_gpu_counts(output) == expected


@pytest.mark.parametrize(
    "output",
    (
        "gpu:a100:not-a-count\n",
        "gpu:a100:4(S:0-1,4-5\n",
        "gpu:a100:4)\n",
        "gpu:a100:4((S:0-1))\n",
        "gpu:a100:4,\n",
        ",gpu:a100:4\n",
        "gpu:a100:4,,mps:100\n",
    ),
)
def test_parse_gpu_counts_rejects_malformed_gpu_resources(output: str) -> None:
    with pytest.raises(SlurmParseError, match="invalid GPU resource"):
        parse_gpu_counts(output)


@pytest.mark.parametrize("state", ("", "CANCELLED by root"))
def test_parse_state_rejects_invalid_spellings(state: str) -> None:
    with pytest.raises(SlurmParseError):
        parse_state(state)


def _make_queue_record(array_task_id: int, state: SchedulerState) -> QueueRecord:
    return QueueRecord(
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=array_task_id),
        state=state,
    )
