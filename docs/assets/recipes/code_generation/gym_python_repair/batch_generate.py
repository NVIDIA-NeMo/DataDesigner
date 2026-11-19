# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and qualify a balanced set of Python repair tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from model_config import API_KEY_ENV, GENERATOR_MODEL, MODEL_BASE_URL

import data_designer.config as dd
from data_designer.interface import DataDesigner

PUBLIC_MISMATCH = "PUBLIC_MISMATCH:"
FAMILIES = {
    "keep_first": "keep the first record for every ID, ignoring subsequent dates",
    "keep_last": "keep the last record for every ID, ignoring which date is newest",
    "ties_keep_last": "select the newest date, but on tied dates keep the last record",
}
STYLES = ["one-pass loop", "mapping-based loop", "two passes", "group records first"]
PROMPT = """Write a deliberately buggy Python starter for a repair exercise.
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
    builder.add_column(dd.LLMTextColumnConfig(name="starter_code", model_alias="generator", prompt=PROMPT))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--repeats", type=int, choices=range(1, len(STYLES) + 1), default=len(STYLES))
    parser.add_argument("--model", default=GENERATOR_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    generate(args.run.resolve(), args.repeats, args.model, args.max_tokens)
