---
date: 2026-04-09
status: in-progress
authors:
  - andreatgretel
---

# Plan: PR Hygiene Automation

Closes #518.

## Problem

External contributors open PRs and never come back - DCO, title, or other required
checks fail and the PR sits indefinitely. The PR template and CONTRIBUTING.md already
ask for linked issues and proper formatting, but nothing enforces it.

## Goals

1. **Linked issue check** - external PRs must reference a triaged issue to merge.
2. **Stale PR cleanup** - remind authors of failing checks, auto-close if unaddressed.
3. **Minimal friction** for collaborators - team members bypass the linked-issue check.

## Non-goals

- No agent or self-hosted runner involvement (plain GitHub Actions on `ubuntu-latest`).
- Not part of the agentic CI plan (plans/472).

---

## Design

### Linked issue check (`pr-linked-issue.yml`)

**Trigger:** `pull_request_target: [opened, edited, synchronize, reopened]` +
`issues: [labeled]` (for re-check when an issue is triaged).

Uses `pull_request_target` so the workflow token can post comments on fork PRs.
Safe because the workflow never checks out or executes PR code.

**Two jobs:**

1. **`check`** (PR events) - validates the PR:
   - Collaborators (`admin`/`write` permission) pass unconditionally.
   - Non-collaborators must have `Fixes #N`/`Closes #N`/`Resolves #N` in the PR
     body, pointing to an existing issue with the `triaged` label.
   - Posts/updates a comment explaining what's missing or confirming success.

2. **`retrigger`** (issue labeled) - when `triaged` is added to an issue, finds
   open PRs referencing it and edits a hidden HTML comment in their body to
   trigger the `edited` event, which re-runs the `check` job.

### Stale PR cleanup (`pr-stale.yml`)

**Trigger:** daily cron (09:00 UTC) + `workflow_dispatch`.

For each open PR with failing checks and no author activity:

| Author type | Reminder | Auto-close |
|-------------|----------|------------|
| Non-collaborator | 7 days | 14 days |
| Collaborator | 14 days | 28 days |

The `keep-open` label prevents auto-close.

Uses `gh api` directly for comment management (looping over multiple PRs).

### Labels

| Label | Purpose |
|-------|---------|
| `triaged` | Maintainer approval gate for external PRs |
| `task` | Auto-label for development-task issue template (was missing) |
| `keep-open` | Prevent stale-PR auto-close |

---

## Deliverables

- [x] Labels created (`triaged`, `task`, `keep-open`)
- [ ] `.github/workflows/pr-linked-issue.yml`
- [ ] `.github/workflows/pr-stale.yml`
- [ ] CONTRIBUTING.md updated with linked-issue requirement
- [ ] `Linked Issue Check` added as required status check on `main` (post-merge)

## Validation

- Open test PR from non-collaborator, verify check blocks without triaged issue.
- Add `triaged` to linked issue, verify re-check passes.
- Verify collaborator PRs pass without a linked issue.
- Run stale workflow via `workflow_dispatch`, verify correct PR targeting.
- Confirm `keep-open` label prevents auto-close.
