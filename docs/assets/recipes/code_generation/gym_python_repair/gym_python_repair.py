# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate buggy Python, collect Gym repair rollouts, and verify the repairs."""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import pandas as pd
import requests
import yaml
from data_designer_gym.config import GymResourceColumnConfig

import data_designer.config as dd
from data_designer.interface import DataDesigner

API_KEY_ENV = "OPENROUTER_API_KEY"
MODEL_BASE_URL = "https://openrouter.ai/api/v1"
GENERATOR_MODEL = "moonshotai/kimi-k3.1-pro"
REPAIR_MODEL = "nvidia/nemotron-3.5-lightning"
ENDPOINT = "http://127.0.0.1:18840"
RECIPE = Path(__file__).resolve().parent
IMAGE = "dd-gym-poc-direct:pi-1.5.2"
WORKSPACE = ("clean.py", "orders.csv", "public_test.py")
REPAIR_PROMPT = """Repair `clean.py`. Its `dedupe(rows)` function takes a list of dictionaries
from `orders.csv`. Return the original record with the newest ISO date in
`updated_at` for each `order_id`, preserving first-seen order across distinct
IDs. On equal dates keep the first record. Keep the function signature, do not
mutate the input, and edit only `clean.py`.

Run `python public_test.py` to reproduce the failure and check your fix.

Run the public test before and after your edit. Once it passes, finish with a brief summary.
The caller will run additional tests separately.
"""

PUBLIC_MISMATCH = "PUBLIC_MISMATCH:"
FAMILIES = {
    "keep_first": "keep the first record for every ID, ignoring subsequent dates",
    "keep_last": "keep the last record for every ID, ignoring which date is newest",
    "ties_keep_last": "select the newest date, but on tied dates keep the last record",
}
STYLES = ["one-pass loop", "mapping-based loop", "two passes", "group records first"]
GENERATION_PROMPT = """Write a deliberately buggy Python starter for a repair exercise.
Implement dedupe(rows), where rows is a list of dictionaries with order_id and updated_at.
Return one original record per order_id in the order the IDs were first seen. The intended
rule is to keep the newest ISO date, keeping the first record when dates tie.
Introduce this bug and only this bug: {{ defect }}.
Use this implementation style if practical: {{ style }}.
Return Python source only, no markdown, prose, inputs/outputs or imports."""
PUBLIC_TEST = """import csv
from copy import deepcopy
from pathlib import Path

from clean import dedupe

with Path(__file__).with_name("orders.csv").open() as data:
    rows = list(csv.DictReader(data))
expected = deepcopy([rows[{winner}], rows[1]])
actual = dedupe(rows)
if actual != expected:
    print(f"PUBLIC_MISMATCH: expected {{expected!r}}, got {{actual!r}}")
    raise SystemExit(1)
print("public example passed")
"""


def execute_sample(folder: Path, entrypoint: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--mount",
            f"type=bind,src={folder},dst=/workspace,readonly",
            "--workdir",
            "/workspace",
            "python:3.12-slim",
            "timeout",
            "-k",
            "1s",
            "15s",
            "python",
            "-B",
            entrypoint,
        ],
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )


def make_cases(repeats: int) -> list[dict[str, str]]:
    return [
        {
            "row_id": f"{family}-{index + 1:02}",
            "family": family,
            "defect": defect,
            "style": style,
        }
        for index, style in enumerate(STYLES[:repeats])
        for family, defect in FAMILIES.items()
    ]


def qualify(row: dict[str, Any], run: Path) -> dict[str, Any]:
    folder = run / "tasks" / row["row_id"]
    folder.mkdir(parents=True)
    result = {**row, "qualification": "missing_source", "qualification_s": 0.0}
    if not isinstance(row["starter_code"], str) or not row["starter_code"].strip():
        return result
    source = row["starter_code"].strip()
    if source.startswith("```") and source.endswith("```") and "\n" in source:
        source = source.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result["starter_code"] = source
    try:
        compile(source, "clean.py", "exec")
    except SyntaxError as exc:
        return {**result, "qualification": "syntax_error", "public_result": str(exc)}
    family = row["family"]
    suffix = row["row_id"][-2:]
    a = f"Z{suffix}"
    b = f"A{suffix}"
    first, last = {
        "keep_first": ("01", "03"),
        "keep_last": ("03", "01"),
        "ties_keep_last": ("03", "03"),
    }[family]
    winner = 2 if family == "keep_first" else 0
    (folder / "clean.py").write_text(source + "\n")
    (folder / "orders.csv").write_text(
        f"order_id,updated_at,status\n{a},2026-10-{first},pending\n{b},2026-10-02,packed\n{a},2026-10-{last},shipped\n"
    )
    (folder / "public_test.py").write_text(PUBLIC_TEST.format(winner=winner))
    started = perf_counter()
    try:
        test = execute_sample(folder, "public_test.py")
    except subprocess.TimeoutExpired:
        return {
            **result,
            "qualification": "timeout",
            "qualification_s": perf_counter() - started,
        }
    status = "runtime_error"
    if test.returncode == 0:
        status = "public_pass"
    elif test.returncode == 1 and test.stdout.startswith(PUBLIC_MISMATCH):
        status = "qualified"
    elif test.returncode in (124, 137):
        status = "timeout"
    return {
        **result,
        "qualification": status,
        "qualification_s": perf_counter() - started,
        "public_result": test.stdout.strip() or test.stderr.strip(),
    }


def generate(run: Path, repeats: int, model: str = GENERATOR_MODEL, max_tokens: int = 4096) -> None:
    if not os.environ.get(API_KEY_ENV):
        raise SystemExit(f"Set {API_KEY_ENV} before generation.")
    run.mkdir(parents=True, exist_ok=False)
    cases = make_cases(repeats)
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="generator",
                model=model,
                provider="openrouter",
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.5, max_tokens=max_tokens, max_parallel_requests=4
                ),
            )
        ],
    ).with_seed_dataset(dd.DataFrameSeedSource(df=pd.DataFrame(cases)), sampling_strategy="ordered")
    builder.add_column(dd.LLMTextColumnConfig(name="starter_code", model_alias="generator", prompt=GENERATION_PROMPT))
    start = perf_counter()
    designer = DataDesigner(
        artifact_path=run / "generation",
        model_providers=[dd.ModelProvider(name="openrouter", endpoint=MODEL_BASE_URL, api_key=API_KEY_ENV)],
    )
    designer.set_run_config(dd.RunConfig(otel_metrics_port=None))
    sampled = designer.create(builder, num_records=len(cases)).load_dataset()
    generation_s = perf_counter() - start
    starters = {r["row_id"]: r["starter_code"] for r in sampled.to_dict("records")}
    rows = [qualify({**case, "starter_code": starters.get(case["row_id"])}, run) for case in cases]
    (run / "stage1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run / "generation.json").write_text(
        json.dumps(
            {
                "target_rows": len(cases),
                "generated_rows": len(starters),
                "generation_s": generation_s,
                "model": model,
                "max_tokens": max_tokens,
            },
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(rows).groupby(["family", "qualification"]).size().to_string())


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
                        "input": [{"role": "user", "content": REPAIR_PROMPT}],
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
        shutil.copyfile(RECIPE / "verify.py", folder / "verify.py")
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


@contextmanager
def gym_servers(run: Path, gym: Path, tokens: list[str]) -> Iterator[None]:
    with (run / "gym.log").open("w") as log:
        servers = subprocess.Popen(
            [str(gym / ".venv" / "bin" / "gym"), "env", "start", *tokens],
            cwd=gym,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "PATH": str(gym / ".venv" / "bin") + os.pathsep + os.environ["PATH"],
                "DD_GYM_POC_SANDBOX_MODEL_HOST": "host.docker.internal",
                "PYTHONUNBUFFERED": "1",
            },
        )
        try:
            deadline = time.monotonic() + 240
            while servers.poll() is None and time.monotonic() < deadline:
                try:
                    with urlopen(f"{ENDPOINT}/openapi.json", timeout=2):
                        break
                except (TimeoutError, URLError):
                    time.sleep(1)
            else:
                raise RuntimeError(f"Gym did not start. See {run / 'gym.log'}")
            yield
        finally:
            if servers.poll() is None:
                os.killpg(servers.pid, signal.SIGINT)
                try:
                    servers.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(servers.pid, signal.SIGKILL)
                    servers.wait()


def repair(run: Path, gym: Path, concurrency: int) -> None:
    if not os.environ.get(API_KEY_ENV):
        raise SystemExit(f"Set {API_KEY_ENV} before starting Gym.")
    label = uuid4().hex
    overlay = run / "batch-gym.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "use_absolute_ip": False,
                "skip_venv_if_present": True,
                "head_server": {"host": "127.0.0.1", "port": 18839},
                "policy_base_url": MODEL_BASE_URL,
                "policy_api_key": f"${{oc.env:{API_KEY_ENV}}}",
                "policy_model_name": REPAIR_MODEL,
                "policy_model": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 18841}}},
                "pi_agent": {
                    "responses_api_agents": {
                        "pi_agent": {
                            "host": "127.0.0.1",
                            "port": 18840,
                            "pi_version": "1.5.2",
                            "sandbox_runtime_dir": "/opt/pi-runtime",
                            "context_window": 32768,
                            "max_output_tokens": 4096,
                            "sandbox_bash_timeout_seconds": 15,
                            "concurrency": concurrency,
                            "sandbox_provider": "sandbox",
                            "sandbox_config": {
                                "image": IMAGE,
                                "workdir": "/workspace",
                                "ttl_s": 240,
                                "resources": {"cpu": 1, "memory_mib": 1024},
                                "provider_options": {"run_args": ["--label", f"dd-gym-showcase={label}"]},
                            },
                        }
                    }
                },
                "sandbox": {
                    "docker": {
                        "create": {
                            "cap_drop": ["ALL"],
                            "security_opt": ["no-new-privileges"],
                            "pids_limit": 256,
                        }
                    }
                },
            }
        )
    )
    paths = [
        gym / "responses_api_agents/pi_agent/configs/pi_agent.yaml",
        gym / "responses_api_models/openai_model/configs/openai_model.yaml",
        gym / "nemo_gym/sandbox/providers/docker/configs/docker.yaml",
        overlay,
    ]
    tokens = ["+config_paths=" + json.dumps([str(p) for p in paths], separators=(",", ":"))]
    with gym_servers(run, gym, tokens):
        run_dd(run, ENDPOINT, label, concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)

    generation = commands.add_parser("generate", help="Generate and qualify starters.")
    generation.add_argument("run", type=Path)
    generation.add_argument("--repeats", type=int, choices=range(1, len(STYLES) + 1), default=len(STYLES))
    generation.add_argument("--model", default=GENERATOR_MODEL)
    generation.add_argument("--max-tokens", type=int, default=4096)

    repairs = commands.add_parser("repair", help="Collect and verify repair rollouts.")
    repairs.add_argument("run", type=Path)
    repairs.add_argument("--gym-runtime", type=Path, required=True)
    repairs.add_argument("--concurrency", type=int, choices=range(1, 5), default=3)

    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage == "generate":
        if args.max_tokens < 1:
            generation.error("--max-tokens must be positive")
        generate(run, args.repeats, args.model, args.max_tokens)
    else:
        gym = args.gym_runtime.resolve()
        if not (gym / ".venv" / "bin" / "gym").is_file():
            repairs.error("--gym-runtime must point to the prepared Gym directory")
        if not (run / "stage1.jsonl").is_file():
            repairs.error("run generate first")
        repair(run, gym, args.concurrency)


if __name__ == "__main__":
    main()
