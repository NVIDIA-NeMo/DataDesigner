## Phase directive

This invocation runs the **FIX** phase only.

- The audit phase has already completed in a previous invocation. Its
  report is at `/tmp/audit-{{suite}}.md` and
  `{{memory_path}}/runner-state.json` has the populated `fix_backlog`.
- Execute only the recipe's "Fix phase" section per `_fix-policy.md`.
  Do NOT redo audit work — that is, do NOT re-scan whole packages or
  rebuild `fix_backlog` from scratch. The "no re-scan" rule does NOT
  override the per-candidate re-verification step required by
  `_fix-policy.md` §"Standard fix procedure" step 4.1: when you pick a
  candidate, you MUST re-grep / re-read the specific file or symbol it
  points at to confirm the finding still applies before editing.
  Re-verification of a single candidate is required; re-scanning the
  codebase to discover new findings is forbidden.
- Pick the highest-ranked eligible candidate from `fix_backlog`, apply
  the fix, run the package's tests if applicable, commit, push, and open
  the PR using `gh pr create --body-file`. If the recipe and
  `_fix-policy.md` declare the category batchable, you may add sibling
  entries from the existing `fix_backlog` after re-verifying each one.
  Do not scan for findings that are not already in `fix_backlog`.
- Record the attempt in `attempted_fixes` (whether successful, abandoned,
  or failed through the top-5 fallback) before exiting. Batch PRs record
  one attempt per fixed finding, all pointing to the same PR and branch.
- If no candidate qualifies after trying up to 5 of them, exit cleanly,
  append a short note to `/tmp/audit-{{suite}}.md` describing what was
  tried, and update `attempted_fixes` accordingly. Do NOT open a PR.
- Do NOT delete branches, even on failure (per `_runner.md` and
  `_fix-policy.md`). Leave them for the existing `pr-stale.yml` workflow
  to reap over time.
- Read the recipe in full for context, but treat the audit phase as
  already done.
