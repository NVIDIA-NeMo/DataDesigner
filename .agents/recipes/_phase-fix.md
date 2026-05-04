## Phase directive

This invocation runs the **FIX** phase only.

- The audit phase has already completed in a previous invocation. Its
  report is at `/tmp/audit-{{suite}}.md` and
  `{{memory_path}}/runner-state.json` has the populated `fix_backlog`.
- Execute only the recipe's "Fix phase" section per `_fix-policy.md`.
  Do NOT redo audit work; do NOT re-scan the codebase to rebuild
  findings.
- Pick the highest-ranked eligible candidate from `fix_backlog`, apply
  the fix, run the package's tests if applicable, commit, push, and open
  the PR using `gh pr create --body-file`.
- Record the attempt in `attempted_fixes` (whether successful, abandoned,
  or failed through the top-5 fallback) before exiting.
- If no candidate qualifies after trying up to 5 of them, exit cleanly,
  append a short note to `/tmp/audit-{{suite}}.md` describing what was
  tried, and update `attempted_fixes` accordingly. Do NOT open a PR.
- Do NOT delete branches, even on failure (per `_runner.md` and
  `_fix-policy.md`). Leave them for the existing `pr-stale.yml` workflow
  to reap over time.
- Read the recipe in full for context, but treat the audit phase as
  already done.
