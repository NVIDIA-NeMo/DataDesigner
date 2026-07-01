# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Data Designer quality/cost and diversity experiments through Switchyard."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner

QUALITY_ROUTES = ("dd-weak", "dd-strong", "dd-hinted")
DIVERSITY_ROUTES = ("dd-weak", "dd-mixed")
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


class FinalAnswer(BaseModel):
    answer: str = Field(description="The final answer only, without explanation")


@dataclass(frozen=True)
class QualityTask:
    task_id: str
    difficulty: str
    problem: str
    expected: str


def _derangements(n: int) -> int:
    previous, current = 1, 0
    for value in range(2, n + 1):
        previous, current = current, (value - 1) * (current + previous)
    return current


def quality_tasks() -> list[QualityTask]:
    onto_functions = sum((-1) ** i * math.comb(5, i) * (5 - i) ** 8 for i in range(6))
    crt = next(value for value in range(1, 7 * 11 * 13 + 1) if value % 7 == 3 and value % 11 == 5 and value % 13 == 8)
    coefficient = sum(math.comb(5, j) * math.comb(12, 7 - 2 * j) for j in range(4))
    return [
        QualityTask("easy-uppercase", "easy", "Write 'cobalt blue' in uppercase.", "COBALT BLUE"),
        QualityTask("easy-lowercase", "easy", "Write 'HELLO WORLD' in lowercase.", "hello world"),
        QualityTask(
            "easy-sort",
            "easy",
            "Sort these words alphabetically and separate them with single spaces: pear apple fig banana.",
            "apple banana fig pear",
        ),
        QualityTask("easy-addition", "easy", "Compute 24 plus 19.", "43"),
        QualityTask("easy-binary", "easy", "Convert binary 101101 to decimal.", "45"),
        QualityTask("easy-count", "easy", "How many times does the letter s appear in 'mississippi'?", "4"),
        QualityTask("easy-subtraction", "easy", "Compute 100 minus 37.", "63"),
        QualityTask("easy-product", "easy", "Compute 17 multiplied by 6.", "102"),
        QualityTask("easy-division", "easy", "Compute 144 divided by 12.", "12"),
        QualityTask("easy-successor", "easy", "What integer comes immediately after 999?", "1000"),
        QualityTask("easy-minimum", "easy", "Return the smallest number in: 18, 3, 27, 11.", "3"),
        QualityTask("easy-maximum", "easy", "Return the largest number in: 45, 91, 12, 67.", "91"),
        QualityTask("easy-boolean", "easy", "Is 3 less than 5? Answer true or false.", "true"),
        QualityTask("easy-weekday", "easy", "Which day comes immediately after Monday?", "Tuesday"),
        QualityTask("easy-hex", "easy", "Convert hexadecimal 1F to decimal.", "31"),
        QualityTask("easy-square", "easy", "Compute 13 squared.", "169"),
        QualityTask(
            "hard-onto-functions",
            "hard",
            "How many onto functions are there from an 8-element set to a 5-element set?",
            str(onto_functions),
        ),
        QualityTask(
            "hard-derangements", "hard", "How many derangements are there of 9 distinct objects?", str(_derangements(9))
        ),
        QualityTask(
            "hard-crt",
            "hard",
            "Find the smallest positive integer x with x mod 7 = 3, x mod 11 = 5, and x mod 13 = 8.",
            str(crt),
        ),
        QualityTask(
            "hard-coefficient",
            "hard",
            "Find the coefficient of x^7 in (1+x)^12 (1+x^2)^5.",
            str(coefficient),
        ),
    ]


def normalize_answer(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace(",", ""))


def evaluate_quality(dataset: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in dataset.to_dict(orient="records"):
        response = row.get("response")
        answer = response.get("answer") if isinstance(response, dict) else ""
        correct = normalize_answer(answer) == normalize_answer(row["expected"])
        rows.append({**row, "answer": answer, "correct": correct})

    result: dict[str, Any] = {
        "correct": sum(bool(row["correct"]) for row in rows),
        "total": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "rows": rows,
    }
    for difficulty in ("easy", "hard"):
        subset = [row for row in rows if row["difficulty"] == difficulty]
        result[f"{difficulty}_accuracy"] = sum(bool(row["correct"]) for row in subset) / len(subset)
    return result


def diversity_metrics(texts: list[str]) -> dict[str, float | int]:
    documents = [TOKEN_PATTERN.findall(text.lower()) for text in texts]
    tokens = [token for document in documents for token in document]
    bigrams = [pair for document in documents for pair in zip(document, document[1:], strict=False)]
    similarities = []
    for left, right in combinations(documents, 2):
        left_set, right_set = set(left), set(right)
        union = left_set | right_set
        similarities.append(len(left_set & right_set) / len(union) if union else 1.0)
    normalized_texts = {normalize_answer(text) for text in texts}
    return {
        "records": len(texts),
        "vocabulary_size": len(set(tokens)),
        "distinct_1": round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0,
        "distinct_2": round(len(set(bigrams)) / len(bigrams), 4) if bigrams else 0.0,
        "mean_pairwise_jaccard": round(sum(similarities) / len(similarities), 4) if similarities else 0.0,
        "exact_duplicate_rate": round(1 - len(normalized_texts) / len(texts), 4) if texts else 0.0,
    }


def _model_config(route: str, *, temperature: float, max_tokens: int) -> dd.ModelConfig:
    return dd.ModelConfig(
        alias="routed-model",
        model=route,
        provider="switchyard",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=temperature,
            max_tokens=max_tokens,
            max_parallel_requests=4,
            timeout=75,
        ),
    )


def _designer(base_url: str, artifact_path: Path) -> DataDesigner:
    designer = DataDesigner(
        artifact_path=artifact_path,
        model_providers=[
            dd.ModelProvider(
                name="switchyard",
                endpoint=base_url,
                provider_type="openai",
                api_key="unused",
            )
        ],
    )
    designer.set_run_config(dd.RunConfig(disable_early_shutdown=True, progress_interval=30.0))
    return designer


def _quality_builder(route: str) -> dd.DataDesignerConfigBuilder:
    tasks = quality_tasks()
    builder = dd.DataDesignerConfigBuilder(model_configs=[_model_config(route, temperature=0.0, max_tokens=512)])
    builder.with_seed_dataset(
        dd.DataFrameSeedSource(df=pd.DataFrame([asdict(task) for task in tasks])),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="response",
            model_alias="routed-model",
            prompt=(
                "[ROUTE_HINT={{ difficulty }}]\n"
                "Solve the problem. Return only the final answer in the answer field: {{ problem }}"
            ),
            output_format=FinalAnswer,
        )
    )
    return builder


def _diversity_builder(route: str, records: int) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[_model_config(route, temperature=0.9, max_tokens=100)])
    builder.with_seed_dataset(
        dd.DataFrameSeedSource(df=pd.DataFrame({"sample_id": range(records)})),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="ticket",
            model_alias="routed-model",
            system_prompt="You write realistic, concise customer-support utterances.",
            prompt=(
                "Write one customer support ticket from a frustrated customer whose package is three days late. "
                "Use 18 to 28 words. Do not use a greeting, label, bullet, or quotation marks."
            ),
        )
    )
    return builder


def _route_summary(stats: dict[str, Any], costs: dict[str, Any]) -> dict[str, Any]:
    models = stats.get("models", {})
    tiers = stats.get("tiers", {})
    classifier = stats.get("classifier", {})
    return {
        "model_calls": {model: values.get("calls", 0) for model, values in models.items()},
        "strong_calls": tiers.get("strong", {}).get("calls", 0),
        "weak_calls": tiers.get("weak", {}).get("calls", 0),
        "classifier_calls": classifier.get("total_requests", 0),
        "total_tokens": stats.get("total_tokens", {}).get("total", 0),
        "estimated_cost_usd": costs.get("total_cost", 0.0),
    }


def _run_dataset(
    *,
    client: httpx.Client,
    base_url: str,
    route: str,
    builder: dd.DataDesignerConfigBuilder,
    records: int,
    output_dir: Path,
    experiment: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client.post("/v1/stats/reset").raise_for_status()
    started_at = time.perf_counter()
    results = _designer(base_url, output_dir / "artifacts").create(
        builder,
        num_records=records,
        dataset_name=f"{experiment}-{route}",
    )
    elapsed_seconds = time.perf_counter() - started_at
    dataset = results.load_dataset()
    stats = client.get("/v1/stats").raise_for_status().json()
    costs = client.get("/demo/cost-estimate").raise_for_status().json()
    summary = _route_summary(stats, costs)
    summary["elapsed_seconds"] = round(elapsed_seconds, 2)
    return dataset, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".scratch/switchyard-experiments"))
    parser.add_argument("--summary-path", type=Path, default=Path("demos/switchyard/results.json"))
    parser.add_argument("--diversity-records", type=int, default=24)
    parser.add_argument("--quality-routes", nargs="+", choices=QUALITY_ROUTES, default=QUALITY_ROUTES)
    parser.add_argument("--skip-diversity", action="store_true")
    parser.add_argument("--merge-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = os.environ.get("SWITCHYARD_BASE_URL")
    if not base_url:
        raise RuntimeError("SWITCHYARD_BASE_URL is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.summary_path.read_text()) if args.merge_results and args.summary_path.exists() else {}

    proxy_root = base_url.rstrip("/").removesuffix("/v1")
    with httpx.Client(base_url=proxy_root, timeout=180.0) as client:
        client.get("/health").raise_for_status()
        config = client.get("/demo/config").raise_for_status().json()
        quality: dict[str, Any] = previous.get("quality", {}).get("routes", {})
        for route in args.quality_routes:
            dataset, route_summary = _run_dataset(
                client=client,
                base_url=base_url,
                route=route,
                builder=_quality_builder(route),
                records=len(quality_tasks()),
                output_dir=args.output_dir,
                experiment="quality",
            )
            metrics = evaluate_quality(dataset)
            rows = metrics.pop("rows")
            route_summary.update(metrics)
            route_summary["cost_per_correct_row_usd"] = (
                round(route_summary["estimated_cost_usd"] / route_summary["correct"], 6)
                if route_summary["correct"]
                else None
            )
            quality[route] = route_summary
            pd.DataFrame(rows).to_json(
                args.output_dir / f"quality-{route}.jsonl",
                orient="records",
                lines=True,
            )

        diversity: dict[str, Any] = previous.get("diversity", {}).get("routes", {})
        for route in () if args.skip_diversity else DIVERSITY_ROUTES:
            dataset, route_summary = _run_dataset(
                client=client,
                base_url=base_url,
                route=route,
                builder=_diversity_builder(route, args.diversity_records),
                records=args.diversity_records,
                output_dir=args.output_dir,
                experiment="diversity",
            )
            texts = [str(value) for value in dataset["ticket"].dropna()]
            route_summary.update(diversity_metrics(texts))
            diversity[route] = route_summary
            dataset.to_json(
                args.output_dir / f"diversity-{route}.jsonl",
                orient="records",
                lines=True,
            )

    quality = {route: quality[route] for route in QUALITY_ROUTES if route in quality}
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": config,
        "quality": {"task_count": len(quality_tasks()), "routes": quality},
        "diversity": {"record_count": args.diversity_records, "routes": diversity},
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
