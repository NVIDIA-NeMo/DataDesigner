---
name: pr-review
description: Review a pull request and post findings as a PR comment
trigger: pull_request
tool: claude-code
timeout_minutes: 15
max_turns: 30
permissions:
  checks: write
  contents: read
  pull-requests: write
---

# PR Review

Review pull request #{{pr_number}} using the `review-code` skill.

## Instructions

1. Run `/review-code {{pr_number}}`
2. The skill writes the review to `/tmp/review-{{pr_number}}.md`. If it does
   not, save the review output there yourself.
3. Before finishing, read `/tmp/review-{{pr_number}}.md` and verify it contains
   a valid review (Summary, Findings, Verdict sections). If the file is empty
   or malformed, write a brief "Review could not be completed" note to the file
   instead.

## Constraints

- Do NOT post the review to GitHub yourself. The workflow handles posting via
  `gh pr comment --body-file`.
- Do NOT approve or request changes on the PR.
- If the diff is extremely large (>100 changed files), focus on the most
  critical files and note that a full review was not feasible in a single pass.
- If the PR only changes docs/markdown, focus on accuracy, broken links, and
  consistency with code. Skip linting.
- If the PR only changes files under `plans/`, focus on: completeness of the
  plan (gaps, missing phases), feasibility of the proposed approach, alignment
  with existing architecture (check AGENTS.md), and whether open questions are
  identified. Skip linting and code-style checks.
