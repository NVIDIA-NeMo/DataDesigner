# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MARKERS = (
    "gate_violation",
    "lockfile_checkout_failed",
    "lockfile_missing",
    "lockfile_not_committed",
    "lockfile_verification_failed",
)


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

    groups_parser = subparsers.add_parser("groups")
    groups_parser.add_argument("--state", type=Path, required=True)
    groups_parser.add_argument("--prior", type=Path, required=True)

    abandon_parser = subparsers.add_parser("abandon")
    abandon_parser.add_argument("--state", type=Path, required=True)
    abandon_parser.add_argument("--finding-ids", required=True)
    abandon_parser.add_argument("--marker", choices=MARKERS, required=True)

    args = parser.parse_args()
    if args.command == "groups":
        print(json.dumps(new_open_attempt_groups(args.state, args.prior), separators=(",", ":")))
        return

    finding_ids = json.loads(args.finding_ids)
    if not isinstance(finding_ids, list) or not all(isinstance(value, str) for value in finding_ids):
        parser.error("--finding-ids must be a JSON list of strings")
    abandon_open_attempts(args.state, finding_ids, args.marker)


if __name__ == "__main__":
    main()
