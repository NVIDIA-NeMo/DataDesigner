## Phase directive

This invocation runs the **AUDIT** phase only.

- Execute the audit steps from the recipe and write the report to
  `/tmp/audit-{{suite}}.md`.
- Update `{{memory_path}}/runner-state.json` with detected findings,
  including `fix_backlog` entries per `_fix-policy.md` (populated BEFORE
  applying the `known_issues` filter to the report, so fixable findings
  persist across runs even when their report row is suppressed).
- Do NOT attempt any fix. Do NOT create any branches, commits, or PRs.
- Do NOT modify any files outside `{{memory_path}}/` and the report file
  `/tmp/audit-{{suite}}.md` itself.
- A separate invocation will run the FIX phase if `fix_backlog` has
  eligible candidates and the suite has a fix phase.
- Read the recipe in full for context; the "Fix phase" section informs
  which finding categories should populate `fix_backlog`, but you must
  not act on them in this invocation.
