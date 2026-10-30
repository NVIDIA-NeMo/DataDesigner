# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

MARKERS = (
    "gate_violation",
    "lockfile_checkout_failed",
    "lockfile_missing",
    "lockfile_not_committed",
    "lockfile_verification_failed",
)


def reconcile_open_attempts(state_path: Path, repo: str) -> None:
    state = json.loads(state_path.read_text())
    outcomes: dict[int, str] = {}
    for entry in state.get("attempted_fixes") or []:
        attempts = entry.get("attempts") or []
        if not attempts or attempts[-1].get("outcome") != "open":
            continue
        attempt = attempts[-1]
        pr_number = attempt.get("pr_number")
        if pr_number is None:
            continue
        if pr_number not in outcomes:
            result = subprocess.run(
                ["gh", "pr", "view", str(pr_number), "--repo", repo, "--json", "state"],
                check=True,
                capture_output=True,
                text=True,
            )
            outcome = json.loads(result.stdout)["state"].lower()
            if outcome not in {"open", "closed", "merged"}:
                raise ValueError(f"Unexpected PR state: {outcome}")
            outcomes[pr_number] = outcome
        attempt["outcome"] = outcomes[pr_number]

    temporary_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(state, indent=2))
    os.replace(temporary_path, state_path)


def new_open_attempt_groups(state_path: Path, prior_path: Path) -> list[dict[str, Any]]:
    state = json.loads(state_path.read_text())
    prior = json.loads(prior_path.read_text())
    prior_counts = {entry["id"]: entry["n"] for entry in prior}
    groups: dict[tuple[int | None, str | None], dict[str, Any]] = {}

    for entry in state.get("attempted_fixes") or []:
        attempts = entry.get("attempts") or []
        if not attempts or attempts[-1].get("outcome") != "open":
            continue
        if len(attempts) <= prior_counts.get(entry["id"], 0):
            continue

        attempt = attempts[-1]
        key = (attempt.get("pr_number"), attempt.get("branch"))
        group = groups.setdefault(
            key,
            {
                "pr_number": attempt.get("pr_number"),
                "branch": attempt.get("branch"),
                "finding_ids": [],
            },
        )
        group["finding_ids"].append(entry["id"])

    return list(groups.values())


def abandon_open_attempts(state_path: Path, finding_ids: list[str], marker: str) -> None:
    if marker not in MARKERS:
        raise ValueError(f"Unsupported marker: {marker}")

    state = json.loads(state_path.read_text())
    selected = set(finding_ids)
    updated = set()
    for entry in state.get("attempted_fixes") or []:
        if entry.get("id") not in selected:
            continue
        attempts = entry.get("attempts") or []
        if attempts and attempts[-1].get("outcome") == "open":
            attempts[-1]["outcome"] = "abandoned"
            attempts[-1][marker] = True
            updated.add(entry["id"])

    if updated != selected:
        missing = ", ".join(sorted(selected - updated))
        raise ValueError(f"Open attempts not found: {missing}")

    temporary_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(state, indent=2))
    os.replace(temporary_path, state_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage agentic CI attempted-fix groups")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("--state", type=Path, required=True)
    reconcile_parser.add_argument("--repo", required=True)

    groups_parser = subparsers.add_parser("groups")
    groups_parser.add_argument("--state", type=Path, required=True)
    groups_parser.add_argument("--prior", type=Path, required=True)

    abandon_parser = subparsers.add_parser("abandon")
    abandon_parser.add_argument("--state", type=Path, required=True)
    abandon_parser.add_argument("--finding-ids", required=True)
    abandon_parser.add_argument("--marker", choices=MARKERS, required=True)

    args = parser.parse_args()
    if args.command == "reconcile":
        reconcile_open_attempts(args.state, args.repo)
        return
    if args.command == "groups":
        print(json.dumps(new_open_attempt_groups(args.state, args.prior), separators=(",", ":")))
        return

    finding_ids = json.loads(args.finding_ids)
    if not isinstance(finding_ids, list) or not all(isinstance(value, str) for value in finding_ids):
        parser.error("--finding-ids must be a JSON list of strings")
    abandon_open_attempts(args.state, finding_ids, args.marker)


if __name__ == "__main__":
    main()
