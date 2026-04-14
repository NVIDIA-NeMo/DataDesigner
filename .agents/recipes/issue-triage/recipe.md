---
name: issue-triage
description: Weekly triage of open issues and PRs - classify, verify, detect staleness, duplicates, and cross-reference
trigger: schedule
tool: claude-code
timeout_minutes: 15
max_turns: 30
permissions:
  contents: read
  issues: write
  pull-requests: read
---

# Repository Triage

Triage all open issues and pull requests in this repository, then post a
combined report to the tracking issue.

## Instructions

### 1. Gather data

Collect all open issues, open PRs, and recent merge activity:

```bash
# All open issues with metadata
gh issue list --state open --limit 200 \
  --json number,title,state,createdAt,updatedAt,labels,assignees,author,body

# All open PRs with metadata
gh pr list --state open --limit 200 \
  --json number,title,state,createdAt,updatedAt,labels,author,headRefName,body

# Recently merged PRs (last 60 days) to cross-reference
gh pr list --state merged --limit 100 \
  --json number,title,headRefName,body,mergedAt

# PR check status for open PRs
for pr in $(gh pr list --state open --json number --jq '.[].number'); do
  echo "=== PR #${pr} ==="
  gh pr checks "$pr" --json name,state --jq '[.[] | select(.state == "FAILURE" or .state == "ERROR")] | length'
done
```

### 2. Triage issues

For each open issue, determine:

**Classification** (pick one):
- `bug` - something is broken
- `feature` - new capability or enhancement
- `chore` - maintenance, CI, docs, refactoring
- `discussion` - needs design input or decision before work starts

**Staleness** (based on last update, today's date, and activity):
- `active` - updated within the last 14 days
- `aging` - updated 14-30 days ago
- `stale` - no update for 30+ days

**Verification** - check if the issue has been addressed:
- Search merged PRs for closing keywords (`Fixes #N`, `Closes #N`, `Resolves #N`)
  referencing this issue
- Search merged PR titles and branches for keywords matching the issue
- If a merged PR appears to fix the issue, flag it as `potentially resolved`
- If there is an open PR linked to the issue, note the PR number

**Labels as signals** - issues with `needs-attention` were flagged by the stale
PR workflow because their linked PR was auto-closed. Always include these in the
"Action needed" section.

**Duplicates / related** - flag issues that overlap in scope or description.

### 3. Triage PRs

For each open PR, determine:

**Health flags** (check all that apply):
- `no-issue` - PR body has no `Fixes/Closes/Resolves #N` reference (external
  contributors only - collaborators are exempt)
- `issue-closed` - PR links to an issue that is already closed (by another PR
  or manually)
- `checks-failing` - PR has failing CI checks
- `stale` - no author activity (push or comment) for 14+ days with failing
  checks
- `duplicate-fix` - another open PR references the same issue

**Cross-reference** - for each PR that references an issue:
- Verify the linked issue exists and is open
- Check if another open or merged PR also references the same issue
- If two open PRs fix the same issue, flag both as `duplicate-fix`

### 4. Build the report

Write the combined report to `/tmp/issue-triage-report.md` using this format:

```markdown
<!-- agentic-ci-issue-triage -->
## Repository Triage Report

**Run date:** YYYY-MM-DD
**Open issues:** N | **Open PRs:** N

---

### Issues: action needed

Issues that need maintainer attention (potentially resolved, stale with no
assignee, possible duplicates, needs-attention label).

| # | Title | Category | Staleness | Flag | Notes |
|---|-------|----------|-----------|------|-------|

### Issues: active work

Issues with assignees or linked open PRs.

| # | Title | Category | Assignee | PR | Last updated |
|---|-------|----------|----------|-----|-------------|

### Issues: backlog

Remaining open issues, ordered by staleness (most stale first).

| # | Title | Category | Staleness | Last updated |
|---|-------|----------|-----------|-------------|

---

### PRs: action needed

PRs with health flags that need maintainer attention.

| # | Title | Author | Flags | Notes |
|---|-------|--------|-------|-------|

### PRs: healthy

Open PRs with no flags.

| # | Title | Author | Linked issue | Last updated |
|---|-------|--------|-------------|-------------|

---

### Summary

**Issues:**
- N triaged, N flagged for action, N active, N backlog
- Flags: X potentially resolved, Y stale, Z duplicates

**PRs:**
- N triaged, N flagged
- Flags: X no linked issue, Y checks failing, Z stale, W duplicate fixes
```

### 5. Post the report

Find the tracking issue number from the `ISSUE_TRIAGE_TRACKING_ISSUE`
environment variable. Find the last comment by `github-actions[bot]` that
contains `<!-- agentic-ci-issue-triage -->` and note its ID.

- If a previous comment exists, **edit it in place** using
  `gh api -X PATCH repos/{owner}/{repo}/issues/comments/{id}`.
- If no previous comment exists, post a new comment using `gh issue comment`.

```bash
# Edit existing comment
gh api -X PATCH "repos/${GITHUB_REPOSITORY}/issues/comments/${COMMENT_ID}" \
  -f body="$(cat /tmp/issue-triage-report.md)"

# Or post new comment
gh issue comment "$TRACKING_ISSUE" --body-file /tmp/issue-triage-report.md
```

## Constraints

- **Read-only triage.** Do not close, label, or modify any issues or PRs. The
  report is for maintainers to act on.
- **Do not post the report yourself if you cannot find the tracking issue.**
  Write the report to `/tmp/issue-triage-report.md` and stop. The workflow
  will handle fallback posting.
- **Stay concise.** Notes columns should be one sentence max. Link to the
  relevant PR, issue, or duplicate - don't explain the fix.
- **Cost awareness.** Do not read full issue/PR bodies unless needed to
  determine duplicates or verify cross-references. The metadata from
  `gh issue list` and `gh pr list` is enough for most checks.
