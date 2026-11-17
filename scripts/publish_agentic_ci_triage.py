# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

COMMENT_LIMIT = 60_000
REPORT_MARKER = re.compile(r"^<!-- agentic-ci-issue-triage:\d+/\d+ -->")


def report_parts(report: str, date: str) -> list[str]:
    """Split a report at section boundaries and add stable comment markers."""
    report = report.strip()
    if not report:
        raise ValueError("Triage report is empty.")

    chunks = [""]
    for section in re.split(r"(?m)(?=^### )", report):
        # Reserve room for the heading and comment marker.
        if len(section) > COMMENT_LIMIT - 200:
            raise ValueError("A triage report section is too large to publish.")
        if len(chunks[-1]) + len(section) > COMMENT_LIMIT - 200:
            chunks.append(section)
        else:
            chunks[-1] += section

    total = len(chunks)
    return [
        f"<!-- agentic-ci-issue-triage:{index}/{total} -->\n## Repository Triage Report ({date})\n\n{chunk.strip()}\n"
        for index, chunk in enumerate(chunks, start=1)
    ]


def publish_report(repo: str, issue: int, parts: list[str]) -> None:
    """Update the bot's existing report comments and reconcile their count."""
    endpoint = f"repos/{repo}/issues/{issue}/comments"
    pages = json.loads(subprocess.check_output(["gh", "api", "--paginate", "--slurp", endpoint], text=True))
    existing = sorted(
        comment["id"]
        for page in pages
        for comment in page
        if (comment.get("user") or {}).get("login") == "github-actions[bot]"
        and REPORT_MARKER.match(comment.get("body") or "")
    )

    for index, body in enumerate(parts):
        if index < len(existing):
            method, target = "PATCH", f"repos/{repo}/issues/comments/{existing[index]}"
        else:
            method, target = "POST", endpoint
        subprocess.run(
            ["gh", "api", "--method", method, target, "--input", "-"],
            input=json.dumps({"body": body}),
            text=True,
            stdout=subprocess.DEVNULL,
            check=True,
        )

    for comment_id in existing[len(parts) :]:
        subprocess.run(
            ["gh", "api", "--method", "DELETE", f"repos/{repo}/issues/comments/{comment_id}"],
            stdout=subprocess.DEVNULL,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the generated repository triage report.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue", required=True, type=int)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    if args.issue < 1:
        parser.error("Issue number must be positive.")
    date = datetime.now(UTC).date().isoformat()
    publish_report(args.repo, args.issue, report_parts(args.report.read_text(encoding="utf-8"), date))


if __name__ == "__main__":
    main()
