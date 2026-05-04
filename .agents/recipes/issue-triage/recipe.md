---
name: issue-triage
description: Weekly triage of open issues and PRs - decision-ready report organized by recommended action
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
decision-ready report to the tracking issue. The report is organized by
**recommended action** so a maintainer can resolve flagged items without
opening each one.

## Instructions

### 1. Gather data

Collect all open issues, open PRs, and recent merge activity:

```bash
gh issue list --state open --limit 200 \
  --json number,title,state,createdAt,updatedAt,labels,assignees,author,body

gh pr list --state open --limit 200 \
  --json number,title,state,createdAt,updatedAt,labels,author,headRefName,body

gh pr list --state merged --limit 100 \
  --json number,title,headRefName,body,mergedAt

# Failing-check counts for open PRs
for pr in $(gh pr list --state open --json number --jq '.[].number'); do
  FAILING=$(gh pr checks "$pr" --json name,state \
    --jq '[.[] | select(.state == "FAILURE" or .state == "ERROR")] | length')
  echo "${pr} ${FAILING}"
done
```

### 2. Decide an action for every flagged item

For each open issue and PR, decide whether it needs maintainer action and, if
so, which **action bucket** it belongs in. Buckets are exclusive — every
flagged item appears under exactly one heading.

Buckets and the criteria for each:

| Bucket | Apply when |
|--------|-----------|
| `Close as resolved` | A merged PR closes the issue via `Fixes/Closes/Resolves #N`, OR a merged PR's title/branch/body strongly indicates it addressed the issue. The issue is still open. |
| `Close as duplicate` | An older open issue covers the same scope. Pick the older issue as the canonical one. |
| `Needs maintainer decision` | Issue labeled `discussion`, design-input items with no clear scope, or items labeled `needs-attention` (flagged by the stale-PR workflow because their linked PR was auto-closed). |
| `Ready for assignment` | Well-scoped issue, no assignee, no linked open PR, not stale (updated within 30 days). Brief enough that someone could pick it up today. |
| `Stuck PR` | Open PR with one or more failing checks, OR no author activity (push/comment) for 14+ days. |
| `Duplicate PRs` | Two or more open PRs reference the same issue (`Fixes/Closes/Resolves #N`). |
| `Stale, consider closing` | 60+ days since last activity, no assignee, no linked open PR. Older than `Ready for assignment` and without traction. |

Items that don't fit any of the above are **healthy** — count them but do
not list them in the action sections.

Also check `attempted_fixes` in any daily-suite runner-state files (under
`.agentic-ci-state/` if accessible) — findings with two `closed` or
`abandoned` attempts are surfaced in their own section so the maintainer
sees them alongside other action items. (This section may be empty if
those state files are not available in the triage run; that's fine.)

### 3. Build the report

Write the report to `/tmp/issue-triage-report.md`. Each part of a
multi-comment report is a separate file: `/tmp/issue-triage-report-1.md`,
`/tmp/issue-triage-report-2.md`, etc.

Format:

````markdown
<!-- agentic-ci-issue-triage:1/N -->
## Repository Triage Report — YYYY-MM-DD

**Open issues:** N | **Open PRs:** N | **Healthy (no action needed):** N

---

### Close as resolved (M)

| # | Title | Action | Evidence | Rationale |
|---|-------|--------|----------|-----------|
| #123 | ... | Close | Merged in #456 | PR title says "fix ..." matching issue scope |

### Close as duplicate (M)

| # | Title | Action | Evidence | Rationale |
|---|-------|--------|----------|-----------|
| #234 | ... | Close, point to #200 | Overlaps #200 (older) | Both describe the same crash on empty config |

### Needs maintainer decision (M)

| # | Title | Action | Evidence | Rationale |
|---|-------|--------|----------|-----------|
| #345 | ... | Decide direction | `discussion` label, no consensus | Two competing approaches in comments |

### Ready for assignment (M)

| # | Title | Action | Evidence | Rationale |
|---|-------|--------|----------|-----------|
| #456 | ... | Assign | Scope clear, no assignee | One-line repro, fix likely <50 LOC |

### Stuck PR (M)

| # | Title | Action | Evidence | Rationale |
|---|-------|--------|----------|-----------|
| #567 | ... | Nudge author or close | 3 failing checks, 21d since push | DCO + lint failing, author hasn't responded |

### Duplicate PRs (M)

| # | Title | Action | Evidence | Rationale |
|---|-------|--------|----------|-----------|
| #678 / #679 | both fix #500 | Pick one, close other | Both reference #500 | #678 has tests, #679 is simpler |

### Stale, consider closing (M)

| # | Title | Action | Evidence | Rationale |
|---|-------|--------|----------|-----------|
| #789 | ... | Close with note | 87d no activity, no assignee | No traction; linked design discussion went silent |

### Repeatedly-failed fix attempts (M)

(Only emit this section if any items qualify. See `_fix-policy.md` —
two-strike escalation.)

| Finding | Suite | Attempts | Notes |
|---------|-------|----------|-------|
| ... | docs-and-references | 2 closed | Detector may be flagging a false positive |

---

<details>
<summary>Healthy items (M issues, M PRs)</summary>

(One-line summary of each: `#N <title> — <author> — <last update>`. No
action needed; this block is for completeness.)

</details>

### Summary

- N items flagged for action across 7 buckets
- M PRs flagged (X stuck, Y duplicate)
- K healthy items (collapsed above)
````

The marker on the first line (`<!-- agentic-ci-issue-triage:1/N -->`) is
required. If the report fits in one comment, set N = 1.

### 4. Multi-comment split

GitHub issue comments cap at 65,536 characters. Use a 60,000-char per-part
budget to leave room for body manipulations.

Build the parts:

1. Render the full report. If `len(body) <= 60000`, you have one part. Use
   marker `<!-- agentic-ci-issue-triage:1/1 -->`.
2. Otherwise, split on **action-bucket boundaries** (never split a table
   mid-row). Each part starts with its own marker
   `<!-- agentic-ci-issue-triage:i/N -->` and a heading
   `### Triage Report — Part i of N`.
3. Place the summary and `Healthy items` `<details>` block at the end of
   the last part.

### 5. Post the report

Tracking issue number is in `ISSUE_TRIAGE_TRACKING_ISSUE`. List all
existing bot comments containing `agentic-ci-issue-triage:` (in id
order) and reconcile against the new parts:

- For `i in 0..min(len(existing), len(parts))`: PATCH `existing[i]`
  with `parts[i]` (`gh api -X PATCH .../comments/<id> -f body=...`).
- Surplus parts: post via `gh issue comment --body-file`.
- Surplus existing comments: delete via
  `gh api -X DELETE .../comments/<id>`.

This keeps the report a coherent set across runs whether it grows,
shrinks, or stays stable.

### 6. Fallback

If you cannot find the tracking issue or the API calls fail repeatedly,
write the report parts to `/tmp/issue-triage-report-*.md` and stop. The
workflow's fallback step will post `/tmp/issue-triage-report.md` (which,
for a single-part report, is identical to part 1; the workflow falls back
only on single-part output). The workflow does not handle multi-part
fallback — make sure the report fits in one part if you cannot post it
yourself.

## Constraints

- **Read-only triage.** Do not close, label, or modify any issues or PRs.
  The report is for maintainers to act on.
- **Stay concise.** Rationale columns should be one sentence max.
- **No fix authority.** This recipe never opens PRs or commits code. It
  reads, classifies, and posts a report.
- **Cost awareness.** Do not read full issue/PR bodies unless needed to
  determine duplicates, verify cross-references, or decide an action. The
  metadata from `gh issue list` / `gh pr list` is enough for most checks.
