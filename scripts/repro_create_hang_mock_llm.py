# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any
import os

from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.engine.errors import DataDesignerRuntimeError


@dataclass(frozen=True)
class RunArgs:
    max_workers: int
    num_records: int
    sleep_min_s: float
    sleep_max_s: float
    stall_prob: float
    seed: int
    hang_timeout_s: float | None
    dump_stacks_on_timeout: bool
    progress_interval_s: float


def _mock_llm_call(*, rng: random.Random, sleep_min_s: float, sleep_max_s: float, stall_prob: float) -> None:
    if stall_prob > 0.0 and rng.random() < stall_prob:
        # Intentionally block forever to simulate a hung request / deadlock.
        threading.Event().wait()
        return
    time.sleep(rng.uniform(sleep_min_s, sleep_max_s))


def _run_mwe(args: RunArgs) -> int:
    rng = random.Random(args.seed)
    buffer: list[dict[str, Any]] = [{"index": i} for i in range(args.num_records)]
    stop_progress = threading.Event()

    def result_callback(result: dict[str, Any], *, context: dict | None = None) -> None:
        if context is None:
            raise RuntimeError("context is required")
        buffer[int(context["index"])] = result

    def error_callback(exc: Exception, *, context: dict | None = None) -> None:
        if context is None:
            raise RuntimeError("context is required")
        buffer[int(context["index"])] = {"index": int(context["index"]), "error": str(exc)}

    def worker_task(record: dict[str, Any]) -> dict[str, Any]:
        _mock_llm_call(
            rng=rng,
            sleep_min_s=args.sleep_min_s,
            sleep_max_s=args.sleep_max_s,
            stall_prob=args.stall_prob,
        )
        return {**record, "generated": True}

    def progress_loop(executor: ConcurrentThreadExecutor) -> None:
        # Best-effort progress so we can see where it stalls.
        while not stop_progress.is_set():
            submitted = executor.results.submitted_count
            completed = executor.results.completed_count
            last = executor.results.last_completed_at_utc
            print(
                f"[progress] submitted={submitted} completed={completed} "
                f"max_workers={executor.max_workers} last_completed_at_utc={last}",
                file=sys.stderr,
                flush=True,
            )
            stop_progress.wait(args.progress_interval_s)

    try:
        with ConcurrentThreadExecutor(
            max_workers=args.max_workers,
            column_name="mock_llm_column",
            result_callback=result_callback,
            error_callback=error_callback,
            disable_early_shutdown=True,
        ) as executor:
            progress_thread = threading.Thread(target=progress_loop, args=(executor,), daemon=True)
            progress_thread.start()

            for i, record in enumerate(buffer):
                executor.submit(
                    lambda record: worker_task(record),
                    record,
                    context={"index": i},
                    acquire_timeout_s=args.hang_timeout_s,
                    dump_stacks_on_timeout=args.dump_stacks_on_timeout,
                )

            executor.wait_for_completion(
                timeout_s=args.hang_timeout_s,
                dump_stacks_on_timeout=args.dump_stacks_on_timeout,
            )
    except DataDesignerRuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    finally:
        stop_progress.set()

    # Sanity check: ensure all records were updated.
    if any("generated" not in r and "error" not in r for r in buffer):
        print("Unexpected: some records were not updated", file=sys.stderr, flush=True)
        return 3

    print("OK: all records completed", file=sys.stderr, flush=True)
    return 0


def _verify_fix(args: RunArgs, *, repro_wait_s: float) -> int:
    script = __file__
    base_cmd = [
        sys.executable,
        script,
        "run",
        "--max-workers",
        str(args.max_workers),
        "--num-records",
        str(args.num_records),
        "--sleep-min-s",
        str(args.sleep_min_s),
        "--sleep-max-s",
        str(args.sleep_max_s),
        "--stall-prob",
        str(args.stall_prob),
        "--seed",
        str(args.seed),
        "--progress-interval-s",
        str(max(1.0, args.progress_interval_s)),
    ]

    # 1) Reproduce the hang safely by running without a timeout and ensuring it doesn't exit quickly.
    repro = subprocess.Popen(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        repro.communicate(timeout=repro_wait_s)
        print(
            "FAILED: repro process exited unexpectedly; expected it to still be running (hang reproduced).",
            file=sys.stderr,
            flush=True,
        )
        return 4
    except subprocess.TimeoutExpired:
        print("OK: hang reproduced (process still running). Killing repro process...", file=sys.stderr, flush=True)
        repro.kill()
        repro.communicate(timeout=10)

    # 2) Verify the fix: run with a timeout and ensure we fail fast.
    fixed_cmd = base_cmd + ["--hang-timeout-s", str(args.hang_timeout_s or 5.0)]
    fixed = subprocess.run(fixed_cmd, capture_output=True, text=True)
    if fixed.returncode == 0:
        print(
            "FAILED: expected a timeout failure (non-zero exit) but run completed successfully.",
            file=sys.stderr,
            flush=True,
        )
        return 5
    if "Timed out" not in (fixed.stderr or ""):
        print(
            "FAILED: run exited non-zero but did not include expected timeout diagnostics in stderr.",
            file=sys.stderr,
            flush=True,
        )
        print(fixed.stderr, file=sys.stderr, flush=True)
        return 6

    print("OK: timeout diagnostics triggered; fix prevents an infinite hang.", file=sys.stderr, flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal reproduction / diagnosis tool for extreme concurrency hangs in threaded column generation.\n"
            "Uses mock 'LLM calls' with random sleep and an optional probability of tasks stalling forever."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the MWE directly (may hang forever if no timeout is set and stall_prob>0).")
    run.add_argument("--max-workers", type=int, default=1024)
    run.add_argument("--num-records", type=int, default=20000)
    run.add_argument("--sleep-min-s", type=float, default=0.5)
    run.add_argument("--sleep-max-s", type=float, default=1.0)
    run.add_argument("--stall-prob", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--hang-timeout-s", type=float, default=None)
    run.add_argument("--dump-stacks-on-timeout", action="store_true", default=False)
    run.add_argument("--progress-interval-s", type=float, default=5.0)

    verify = sub.add_parser("verify", help="Reproduce the hang safely in a subprocess, then verify the timeout fix.")
    verify.add_argument("--max-workers", type=int, default=1024)
    verify.add_argument("--num-records", type=int, default=20000)
    verify.add_argument("--sleep-min-s", type=float, default=0.5)
    verify.add_argument("--sleep-max-s", type=float, default=1.0)
    verify.add_argument("--stall-prob", type=float, default=0.0005)
    verify.add_argument("--seed", type=int, default=0)
    verify.add_argument("--hang-timeout-s", type=float, default=5.0)
    verify.add_argument("--dump-stacks-on-timeout", action="store_true", default=False)
    verify.add_argument("--progress-interval-s", type=float, default=5.0)
    verify.add_argument("--repro-wait-s", type=float, default=5.0)

    return parser


def main() -> int:
    parser = _build_parser()
    ns = parser.parse_args()

    if ns.sleep_min_s <= 0 or ns.sleep_max_s <= 0 or ns.sleep_min_s > ns.sleep_max_s:
        print("Invalid sleep range.", file=sys.stderr, flush=True)
        return 2
    if ns.max_workers <= 0 or ns.num_records <= 0:
        print("max_workers and num_records must be > 0.", file=sys.stderr, flush=True)
        return 2
    if not (0.0 <= ns.stall_prob <= 1.0):
        print("stall_prob must be in [0.0, 1.0].", file=sys.stderr, flush=True)
        return 2

    args = RunArgs(
        max_workers=int(ns.max_workers),
        num_records=int(ns.num_records),
        sleep_min_s=float(ns.sleep_min_s),
        sleep_max_s=float(ns.sleep_max_s),
        stall_prob=float(ns.stall_prob),
        seed=int(ns.seed),
        hang_timeout_s=(float(ns.hang_timeout_s) if ns.hang_timeout_s is not None else None),
        dump_stacks_on_timeout=bool(ns.dump_stacks_on_timeout),
        progress_interval_s=float(ns.progress_interval_s),
    )

    if ns.cmd == "run":
        return _run_mwe(args)
    if ns.cmd == "verify":
        return _verify_fix(args, repro_wait_s=float(ns.repro_wait_s))
    raise RuntimeError(f"Unknown command: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())


