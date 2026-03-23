"""Utilities for dual-level quality filtering of generated datasets."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _parse_scores(scores: Any) -> dict[str, Any]:
    """Normalize judge outputs to dictionaries."""
    if isinstance(scores, str):
        return json.loads(scores)
    return scores or {}


def filter_high_quality(
    df: pd.DataFrame,
    min_query_feasibility: int = 3,
    min_query_schema_compliance: int = 4,
    min_query_naturalness: int = 3,
    min_trajectory_tool_validity: int = 4,
    min_trajectory_argument_validity: int = 4,
    min_trajectory_completeness: int = 3,
    min_trajectory_efficiency: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter generated data with dual-level quality control.

    Stage 1 checks user-query quality.
    Stage 2 checks trajectory quality.
    Records must pass both stages.
    """
    out = df.copy()
    out["_query_scores"] = out["user_query_judge"].apply(_parse_scores)
    out["_traj_scores"] = out["trajectory_judge"].apply(_parse_scores)

    # Stage 1: user query quality
    query_is_valid = out["_query_scores"].apply(lambda x: x.get("is_valid", False)) == True  # noqa: E712
    query_feasibility_ok = out["_query_scores"].apply(lambda x: x.get("feasibility", 0)) >= min_query_feasibility
    query_schema_ok = out["_query_scores"].apply(lambda x: x.get("schema_compliance", 0)) >= min_query_schema_compliance
    query_natural_ok = out["_query_scores"].apply(lambda x: x.get("naturalness", 0)) >= min_query_naturalness
    query_passed = query_is_valid & query_feasibility_ok & query_schema_ok & query_natural_ok

    # Stage 2: trajectory quality
    traj_is_valid = out["_traj_scores"].apply(lambda x: x.get("is_valid", False)) == True  # noqa: E712
    traj_tool_ok = out["_traj_scores"].apply(lambda x: x.get("tool_validity", 0)) >= min_trajectory_tool_validity
    traj_args_ok = out["_traj_scores"].apply(lambda x: x.get("argument_validity", 0)) >= min_trajectory_argument_validity
    traj_complete_ok = out["_traj_scores"].apply(lambda x: x.get("completeness", 0)) >= min_trajectory_completeness
    traj_efficient_ok = out["_traj_scores"].apply(lambda x: x.get("efficiency", 0)) >= min_trajectory_efficiency
    traj_passed = traj_is_valid & traj_tool_ok & traj_args_ok & traj_complete_ok & traj_efficient_ok

    final_passed = query_passed & traj_passed

    if verbose:
        n = len(out)
        print(f"\n=== Quality Filtering Results ===")
        print(f"Total records: {n}")
        print(f"\nStage 1 (User Query):  {query_passed.sum()}/{n} passed ({query_passed.mean() * 100:.0f}%)")
        print(f"  is_valid: {query_is_valid.sum()} | feasibility>={min_query_feasibility}: {query_feasibility_ok.sum()} "
              f"| schema>={min_query_schema_compliance}: {query_schema_ok.sum()} | naturalness>={min_query_naturalness}: {query_natural_ok.sum()}")
        print(f"\nStage 2 (Trajectory): {traj_passed.sum()}/{n} passed ({traj_passed.mean() * 100:.0f}%)")
        print(f"  is_valid: {traj_is_valid.sum()} | tool_validity>={min_trajectory_tool_validity}: {traj_tool_ok.sum()} "
              f"| arg_validity>={min_trajectory_argument_validity}: {traj_args_ok.sum()} "
              f"| completeness>={min_trajectory_completeness}: {traj_complete_ok.sum()} "
              f"| efficiency>={min_trajectory_efficiency}: {traj_efficient_ok.sum()}")
        print(f"\nFinal: {final_passed.sum()}/{n} passed ({final_passed.mean() * 100:.0f}%)")

    return out[final_passed].drop(columns=["_query_scores", "_traj_scores"]).reset_index(drop=True)


def show_rejection_reasons(df: pd.DataFrame, num_examples: int = 5) -> None:
    """Print example rejection reasons from both judges."""
    query_scores = df["user_query_judge"].apply(_parse_scores)
    traj_scores = df["trajectory_judge"].apply(_parse_scores)

    for label, scores in [("User Query", query_scores), ("Trajectory", traj_scores)]:
        rejected = scores[scores.apply(lambda x: not x.get("is_valid", True))]
        print(f"\n=== {label} Issues ({len(rejected)}/{len(df)} rejected) ===")
        if len(rejected) == 0:
            print("  No issues found.")
            continue
        for i, (idx, s) in enumerate(rejected.head(num_examples).items()):
            print(f"  [{i + 1}] {df.loc[idx, 'user_query'][:100]}...")
            print(f"      Issues: {s.get('issues', 'N/A')}")
