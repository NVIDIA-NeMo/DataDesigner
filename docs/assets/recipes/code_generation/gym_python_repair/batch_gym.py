# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a dedicated Gym Pi agent for the batch experiment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import yaml
from model_config import API_KEY_ENV, MODEL_BASE_URL, REPAIR_MODEL

ENDPOINT = "http://127.0.0.1:18840"
PROJECT = Path(__file__).resolve().parent
IMAGE = "dd-gym-poc-direct:pi-1.5.2"


@contextmanager
def gym_servers(run: Path, tokens: list[str]) -> Iterator[None]:
    with (run / "gym.log").open("w") as log:
        servers = subprocess.Popen(
            ["gym", "env", "start", *tokens],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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


def run_batch(run: Path, dd_python: Path, concurrency: int) -> None:
    if not os.environ.get(API_KEY_ENV):
        raise SystemExit(f"Set {API_KEY_ENV} before starting Gym.")
    label = uuid4().hex
    gym = Path.cwd()
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
    with gym_servers(run, tokens):
        subprocess.run(
            [
                str(dd_python),
                str(PROJECT / "batch_repair.py"),
                str(run),
                "--gym-endpoint",
                ENDPOINT,
                "--label",
                label,
                "--concurrency",
                str(concurrency),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dd-python", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    run_batch(args.run.resolve(), args.dd_python.absolute(), args.concurrency)
