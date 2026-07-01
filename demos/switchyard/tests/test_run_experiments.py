# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from demos.switchyard.run_experiments import diversity_metrics, evaluate_quality, normalize_answer, quality_tasks


def test_quality_tasks_have_stable_expected_answers() -> None:
    tasks = {task.task_id: task for task in quality_tasks()}

    assert len(tasks) == 20
    assert tasks["hard-onto-functions"].expected == "126000"
    assert tasks["hard-derangements"].expected == "133496"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" 126,000 ", "126000"), ("COBALT   BLUE", "cobalt blue")],
)
def test_normalize_answer(value: str, expected: str) -> None:
    assert normalize_answer(value) == expected


def test_evaluate_quality_splits_difficulty() -> None:
    dataset = pd.DataFrame(
        [
            {"difficulty": "easy", "expected": "45", "response": {"answer": "45"}},
            {"difficulty": "hard", "expected": "126000", "response": {"answer": "105000"}},
        ]
    )

    result = evaluate_quality(dataset)

    assert result["accuracy"] == 0.5
    assert result["easy_accuracy"] == 1.0
    assert result["hard_accuracy"] == 0.0


def test_diversity_metrics_reward_varied_text() -> None:
    repeated = diversity_metrics(["late package needs help"] * 3)
    varied = diversity_metrics(
        ["late package needs help", "shipment missing after three days", "parcel still has not arrived"]
    )

    assert varied["distinct_2"] > repeated["distinct_2"]
    assert varied["mean_pairwise_jaccard"] < repeated["mean_pairwise_jaccard"]
    assert varied["exact_duplicate_rate"] == 0.0
