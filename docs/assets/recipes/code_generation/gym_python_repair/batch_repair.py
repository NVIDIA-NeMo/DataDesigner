# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run qualified tasks through direct Pi and a separate verifier column."""

from __future__ import annotations

import argparse
import difflib
import io
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
from typing import Any
from uuid import uuid4

import pandas as pd
import requests
from batch_generate import execute_sample
from data_designer_gym.config import GymResourceColumnConfig

import data_designer.config as dd
from data_designer.interface import DataDesigner

RECIPE = Path(__file__).resolve().parent
WORKSPACE = ("clean.py", "orders.csv", "public_test.py")
PROMPT = (
    (RECIPE / "instruction.md").read_text()
    + """
Run the public test before and after your edit. Once it passes, finish with a brief summary.
The caller will run additional tests separately.
"""
)


class RepairBatch:
    def __init__(self, run: Path, gym_endpoint: str, label: str, rows: list[dict[str, Any]]) -> None:
        self.run = run
        self.gym_endpoint = gym_endpoint
        self.label = label
        self.rows = {row["row_id"]: row for row in rows}
        self.seed_lock = Lock()

    def containers(self) -> set[str]:
        return set(
            subprocess.check_output(
                [
                    "docker",
                    "ps",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"label=dd-gym-showcase={self.label}",
                ],
                text=True,
                timeout=10,
            ).splitlines()
        )

    def gym_post(self, client: requests.Session, route: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
        response = client.post(f"{self.gym_endpoint}{route}", json=body, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"Gym {route.rsplit('/', 1)[-1]} HTTP {response.status_code}: {response.text[:500]}")
        return response.json()

    def stage_workspace(self, container: str, source: Path) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for filename in WORKSPACE:
                data = (source / filename).read_bytes()
                info = tarfile.TarInfo(filename)
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        subprocess.run(
            ["docker", "exec", "-i", "-w", "/workspace", container, "tar", "-xf", "-"],
            input=archive.getvalue(),
            check=True,
            capture_output=True,
            timeout=15,
        )

    def repair(self, row_id: str) -> dict[str, Any]:
        identity = uuid4().hex
        session = {
            "agent_session_id": identity,
            "episode_id": {"rollout_id": identity, "attempt": 0},
        }
        source = self.run / "tasks" / row_id
        evidence = self.run / "episodes" / row_id
        evidence.mkdir(parents=True)
        started = perf_counter()
        result = {"status": "failed", "rollout_id": identity, "cleanup_ok": False}
        container = None
        with requests.Session() as client:
            try:
                # The dedicated Gym server labels every sandbox. Only session creation needs the lock.
                with self.seed_lock:
                    previous = self.containers()
                    self.gym_post(
                        client,
                        "/v1/agent_sessions",
                        {
                            **session,
                            "task_id": {"taskset": "dd-csv-repair", "task_id": row_id},
                            "agent_timeout_sec": 120,
                        },
                        timeout=45,
                    )
                    (container,) = self.containers() - previous
                self.stage_workspace(container, source)
                result["setup_s"] = perf_counter() - started
                activation = perf_counter()
                response = self.gym_post(
                    client,
                    f"/ng-rollout/{identity}/v1/responses",
                    {
                        "input": [{"role": "user", "content": PROMPT}],
                        "metadata": {"agent_timeout_sec": "120"},
                    },
                    timeout=155,
                )
                result["agent_s"] = perf_counter() - activation
                result["agent_status"] = response.get("status")
                result["usage"] = response.get("usage")
                result["tools"] = dict(
                    Counter(item["name"] for item in response.get("output", []) if item.get("type") == "function_call")
                )
                (evidence / "response.json").write_text(json.dumps(response, indent=2) + "\n")
                repaired = subprocess.check_output(
                    ["docker", "exec", container, "cat", "/workspace/clean.py"],
                    text=True,
                    timeout=15,
                )
                (evidence / "clean.py").write_text(repaired)
                (evidence / "repair.diff").write_text(
                    "".join(
                        difflib.unified_diff(
                            (source / "clean.py").read_text().splitlines(keepends=True),
                            repaired.splitlines(keepends=True),
                            fromfile="starter/clean.py",
                            tofile="repair/clean.py",
                        )
                    )
                )
                result["status"] = "completed" if result["agent_status"] == "completed" else "incomplete"
            except (
                OSError,
                RuntimeError,
                ValueError,
                requests.RequestException,
                subprocess.SubprocessError,
            ) as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    close = self.gym_post(client, "/v1/agent_sessions/close", session, timeout=30)
                    (evidence / "close.json").write_text(json.dumps(close, indent=2) + "\n")
                    result["cleanup_ok"] = container is None or container not in self.containers()
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    requests.RequestException,
                    subprocess.SubprocessError,
                ) as exc:
                    result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["repair_s"] = perf_counter() - started
        (evidence / "repair.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def verify(self, row_id: str, repair_status: str | None) -> dict[str, Any]:
        if repair_status != "completed":
            return {
                "result": {
                    "reward": None,
                    "mask_sample": True,
                    "status": "repair_unavailable",
                }
            }
        started = perf_counter()
        folder = self.run / "episodes" / row_id
        shutil.copyfile(RECIPE / "batch_verify.py", folder / "verify.py")
        for name in ("orders.csv", "public_test.py"):
            shutil.copyfile(self.run / "tasks" / row_id / name, folder / name)
        try:
            public = execute_sample(folder, "public_test.py")
            hidden = execute_sample(folder, "verify.py")
            cases = json.loads(hidden.stdout.splitlines()[-1]) if hidden.stdout.strip() else {}
            passed = public.returncode == 0 and hidden.returncode == 0
            status = "passed" if passed else "failed"
            result = {
                "status": status,
                "public_pass": public.returncode == 0,
                "held_out_pass": hidden.returncode == 0,
                "cases": cases,
                "reward": float(passed),
                "mask_sample": False,
                "public_log": public.stdout.strip() or public.stderr.strip(),
                "held_out_log": hidden.stdout.strip() or hidden.stderr.strip(),
            }
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            result = {
                "status": "verifier_error",
                "error": str(exc),
                "reward": None,
                "mask_sample": True,
            }
        result["verify_s"] = perf_counter() - started
        (folder / "verifier.json").write_text(json.dumps(result, indent=2) + "\n")
        return {"result": result}


def run_dd(run: Path, gym_endpoint: str, label: str, concurrency: int) -> None:
    rows = [json.loads(line) for line in (run / "stage1.jsonl").read_text().splitlines()]
    seeds = [row for row in rows if row["qualification"] == "qualified"]
    if not seeds:
        raise SystemExit("No qualified rows.")
    batch = RepairBatch(run, gym_endpoint, label, seeds)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            row_id = body["row_id"]
            if row_id not in batch.rows:
                self.send_error(400, "Unknown row")
                return
            if self.path == "/repair":
                result = batch.repair(row_id)
            elif self.path == "/verify":
                result = batch.verify(row_id, body.get("repair_status"))
            else:
                self.send_error(404)
                return
            encoded = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            address = f"http://127.0.0.1:{server.server_port}"
            builder = dd.DataDesignerConfigBuilder(model_configs=[]).with_seed_dataset(
                dd.DataFrameSeedSource(df=pd.DataFrame(seeds)),
                sampling_strategy="ordered",
            )
            builder.add_column(
                GymResourceColumnConfig(
                    name="rollout",
                    endpoint=address,
                    route="/repair",
                    row_id_column="row_id",
                    request_columns={"row_id": "row_id"},
                    max_parallel_requests=concurrency,
                    timeout=245,
                )
            )
            builder.add_column(
                GymResourceColumnConfig(
                    name="verifier",
                    endpoint=address,
                    route="/verify",
                    row_id_column="row_id",
                    request_columns={
                        "row_id": "row_id",
                        "repair_status": "rollout.response_json.status",
                    },
                    max_parallel_requests=concurrency,
                    timeout=60,
                    response_protocol="gym-episode",
                )
            )
            builder.add_column(
                dd.ExpressionColumnConfig(
                    name="accepted",
                    dtype="bool",
                    expr="{{ verifier.episode_status == 'completed' and verifier.reward == 1 and verifier.mask_sample != true }}",
                )
            )
            start = perf_counter()
            result = DataDesigner(artifact_path=run / "repair").create(builder, num_records=len(seeds)).load_dataset()
            result.to_json(run / "stage2.jsonl", orient="records", lines=True)
            leftovers = len(batch.containers())
            (run / "repair_run.json").write_text(
                json.dumps(
                    {
                        "qualified_rows": len(seeds),
                        "output_rows": len(result),
                        "concurrency": concurrency,
                        "wall_s": perf_counter() - start,
                        "active_batch_containers_after": leftovers,
                    },
                    indent=2,
                )
                + "\n"
            )
            if leftovers:
                raise RuntimeError(f"{leftovers} sandbox containers remain; inspect Gym log.")
        finally:
            server.shutdown()
            worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--gym-endpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    run_dd(args.run.resolve(), args.gym_endpoint, args.label, args.concurrency)
