"""Utilities for dual-level quality filtering in workplace assistant datasets.

This module keeps notebook cells short by moving filtering/debugging logic into
reusable functions with intuitive defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FilterThresholds:
    """Thresholds used by `filter_high_quality`.

    Defaults are tuned for strict schema compliance while keeping naturalness
    and completeness practical for synthetic generation.
    """

    min_query_feasibility: int = 3
    min_query_schema_compliance: int = 4
    min_query_naturalness: int = 3
    min_trajectory_tool_validity: int = 4
    min_trajectory_argument_validity: int = 4
    min_trajectory_completeness: int = 3
    min_trajectory_efficiency: int = 3

    def to_kwargs(self) -> dict[str, int]:
        """Return thresholds in the keyword format expected by filtering API."""
        return {
            "min_query_feasibility": self.min_query_feasibility,
            "min_query_schema_compliance": self.min_query_schema_compliance,
            "min_query_naturalness": self.min_query_naturalness,
            "min_trajectory_tool_validity": self.min_trajectory_tool_validity,
            "min_trajectory_argument_validity": self.min_trajectory_argument_validity,
            "min_trajectory_completeness": self.min_trajectory_completeness,
            "min_trajectory_efficiency": self.min_trajectory_efficiency,
        }


def _parse_scores(scores: Any) -> dict[str, Any]:
    """Normalize judge outputs to dictionaries."""
    if isinstance(scores, str):
        return json.loads(scores)
    return scores or {}


def print_quality_filtering_quickstart() -> None:
    """Print practical usage instructions for notebook users."""
    print("Quality filtering utility loaded.")
    print("\nQuickstart:")
    print("1) Inspect rejection reasons:")
    print("   show_rejection_reasons(results_df, num_examples=3)")
    print("2) Filter with default strict thresholds:")
    print("   filtered_df = filter_high_quality(results_df, verbose=True)")
    print("3) Customize thresholds (optional):")
    print("   thresholds = FilterThresholds(min_query_schema_compliance=5)")
    print("   filtered_df = filter_high_quality(results_df, **thresholds.to_kwargs())")


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
    query_is_valid = out["_query_scores"].apply(lambda x: x.get("is_valid", False)) == True
    query_feasibility_ok = (
        out["_query_scores"].apply(lambda x: x.get("feasibility", 0)) >= min_query_feasibility
    )
    query_schema_ok = (
        out["_query_scores"].apply(lambda x: x.get("schema_compliance", 0))
        >= min_query_schema_compliance
    )
    query_natural_ok = (
        out["_query_scores"].apply(lambda x: x.get("naturalness", 0)) >= min_query_naturalness
    )
    query_passed = query_is_valid & query_feasibility_ok & query_schema_ok & query_natural_ok

    # Stage 2: trajectory quality
    traj_is_valid = out["_traj_scores"].apply(lambda x: x.get("is_valid", False)) == True
    traj_tool_ok = (
        out["_traj_scores"].apply(lambda x: x.get("tool_validity", 0))
        >= min_trajectory_tool_validity
    )
    traj_args_ok = (
        out["_traj_scores"].apply(lambda x: x.get("argument_validity", 0))
        >= min_trajectory_argument_validity
    )
    traj_complete_ok = (
        out["_traj_scores"].apply(lambda x: x.get("completeness", 0))
        >= min_trajectory_completeness
    )
    traj_efficient_ok = (
        out["_traj_scores"].apply(lambda x: x.get("efficiency", 0)) >= min_trajectory_efficiency
    )
    traj_passed = traj_is_valid & traj_tool_ok & traj_args_ok & traj_complete_ok & traj_efficient_ok

    final_passed = query_passed & traj_passed

    if verbose:
        print("=" * 70)
        print("DUAL-LEVEL QUALITY FILTERING RESULTS")
        print("=" * 70)
        print(f"\nTotal records: {len(out)}")

        print(f"\n{'-' * 70}")
        print("STAGE 1: USER QUERY FILTERING")
        print(f"{'-' * 70}")
        print(
            f"  is_valid=True:                 {query_is_valid.sum():4d} / {len(out)} "
            f"({query_is_valid.mean() * 100:5.1f}%)"
        )
        print(
            f"  feasibility >= {min_query_feasibility}:           {query_feasibility_ok.sum():4d} / "
            f"{len(out)} ({query_feasibility_ok.mean() * 100:5.1f}%)"
        )
        print(
            f"  schema_compliance >= {min_query_schema_compliance}:     {query_schema_ok.sum():4d} / "
            f"{len(out)} ({query_schema_ok.mean() * 100:5.1f}%)"
        )
        print(
            f"  naturalness >= {min_query_naturalness}:           {query_natural_ok.sum():4d} / {len(out)} "
            f"({query_natural_ok.mean() * 100:5.1f}%)"
        )
        print("  ----------------------------------------------")
        print(
            f"  PASSED Stage 1:                {query_passed.sum():4d} / {len(out)} "
            f"({query_passed.mean() * 100:5.1f}%)"
        )

        print(f"\n{'-' * 70}")
        print("STAGE 2: TRAJECTORY FILTERING")
        print(f"{'-' * 70}")
        print(
            f"  is_valid=True:                 {traj_is_valid.sum():4d} / {len(out)} "
            f"({traj_is_valid.mean() * 100:5.1f}%)"
        )
        print(
            f"  tool_validity >= {min_trajectory_tool_validity}:        {traj_tool_ok.sum():4d} / "
            f"{len(out)} ({traj_tool_ok.mean() * 100:5.1f}%)"
        )
        print(
            f"  argument_validity >= {min_trajectory_argument_validity}:    {traj_args_ok.sum():4d} / "
            f"{len(out)} ({traj_args_ok.mean() * 100:5.1f}%)"
        )
        print(
            f"  completeness >= {min_trajectory_completeness}:         {traj_complete_ok.sum():4d} / "
            f"{len(out)} ({traj_complete_ok.mean() * 100:5.1f}%)"
        )
        print(
            f"  efficiency >= {min_trajectory_efficiency}:           {traj_efficient_ok.sum():4d} / "
            f"{len(out)} ({traj_efficient_ok.mean() * 100:5.1f}%)"
        )
        print("  ----------------------------------------------")
        print(
            f"  PASSED Stage 2:                {traj_passed.sum():4d} / {len(out)} "
            f"({traj_passed.mean() * 100:5.1f}%)"
        )

        rejected_by_query_only = (~query_passed & traj_passed).sum()
        rejected_by_traj_only = (query_passed & ~traj_passed).sum()
        rejected_by_both = (~query_passed & ~traj_passed).sum()

        print(f"\n{'-' * 70}")
        print("REJECTION BREAKDOWN")
        print(f"{'-' * 70}")
        print(f"  Rejected by Query Judge only:      {rejected_by_query_only:4d}")
        print(f"  Rejected by Trajectory Judge only: {rejected_by_traj_only:4d}")
        print(f"  Rejected by BOTH judges:           {rejected_by_both:4d}")

        print(f"\n{'=' * 70}")
        print(f"FINAL RESULT: {final_passed.sum()} / {len(out)} passed ({final_passed.mean() * 100:.1f}%)")
        print("=" * 70)

    return (
        out[final_passed]
        .drop(columns=["_query_scores", "_traj_scores"])
        .reset_index(drop=True)
    )


def show_rejection_reasons(
    df: pd.DataFrame,
    num_examples: int = 5,
    query_schema_threshold: int = 4,
    trajectory_argument_threshold: int = 4,
) -> None:
    """Print example rejection reasons from both judges."""
    out = df.copy()
    out["_query_scores"] = out["user_query_judge"].apply(_parse_scores)
    out["_traj_scores"] = out["trajectory_judge"].apply(_parse_scores)

    # Stage 1: user query rejections
    query_invalid = out[out["_query_scores"].apply(lambda x: not x.get("is_valid", True))]
    query_schema_issues = out[
        out["_query_scores"].apply(
            lambda x: x.get("schema_compliance", 5) < query_schema_threshold
        )
    ]

    print(f"\n{'=' * 70}")
    print("STAGE 1: USER QUERY ISSUES")
    print("=" * 70)

    if len(query_invalid) > 0:
        print(f"\n[INVALID QUERIES] ({len(query_invalid)} total)")
        for i, (_, row) in enumerate(query_invalid.head(num_examples).iterrows()):
            scores = row["_query_scores"]
            print(f"\n  [{i + 1}] Query: {row['user_query'][:80]}...")
            print(
                f"      Feasibility: {scores.get('feasibility', 'N/A')}/5 | "
                f"Schema: {scores.get('schema_compliance', 'N/A')}/5"
            )
            print(f"      Issues: {scores.get('issues', 'N/A')}")

    schema_only = query_schema_issues[~query_schema_issues.index.isin(query_invalid.index)]
    if len(schema_only) > 0:
        print(f"\n[SCHEMA COMPLIANCE ISSUES] ({len(schema_only)} additional)")
        for i, (_, row) in enumerate(schema_only.head(num_examples).iterrows()):
            scores = row["_query_scores"]
            print(f"\n  [{i + 1}] Query: {row['user_query'][:80]}...")
            print(f"      Schema Compliance: {scores.get('schema_compliance', 'N/A')}/5")
            print(f"      Issues: {scores.get('issues', 'N/A')}")

    if len(query_invalid) == 0 and len(schema_only) == 0:
        print("\n  No user query issues found!")

    # Stage 2: trajectory rejections
    traj_invalid = out[out["_traj_scores"].apply(lambda x: not x.get("is_valid", True))]
    traj_arg_issues = out[
        out["_traj_scores"].apply(
            lambda x: x.get("argument_validity", 5) < trajectory_argument_threshold
        )
    ]

    print(f"\n{'=' * 70}")
    print("STAGE 2: TRAJECTORY ISSUES")
    print("=" * 70)

    if len(traj_invalid) > 0:
        print(f"\n[INVALID TRAJECTORIES] ({len(traj_invalid)} total)")
        for i, (_, row) in enumerate(traj_invalid.head(num_examples).iterrows()):
            scores = row["_traj_scores"]
            print(f"\n  [{i + 1}] Query: {row['user_query'][:80]}...")
            print(
                f"      Tool Validity: {scores.get('tool_validity', 'N/A')}/5 | "
                f"Arg Validity: {scores.get('argument_validity', 'N/A')}/5"
            )
            print(f"      Issues: {scores.get('issues', 'N/A')}")

    arg_only = traj_arg_issues[~traj_arg_issues.index.isin(traj_invalid.index)]
    if len(arg_only) > 0:
        print(f"\n[ARGUMENT VALIDITY ISSUES] ({len(arg_only)} additional)")
        for i, (_, row) in enumerate(arg_only.head(num_examples).iterrows()):
            scores = row["_traj_scores"]
            print(f"\n  [{i + 1}] Query: {row['user_query'][:80]}...")
            print(f"      Argument Validity: {scores.get('argument_validity', 'N/A')}/5")
            print(f"      Issues: {scores.get('issues', 'N/A')}")

    if len(traj_invalid) == 0 and len(arg_only) == 0:
        print("\n  No trajectory issues found!")

    print(f"\n{'=' * 70}\n")
