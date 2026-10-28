# Agentic CI Fix Policy

Prepended to every daily-suite recipe alongside `_runner.md`. Defines what
"open a PR" means for these recipes and the rules that apply across all of
them. Each suite recipe declares only its eligible finding categories, its
branch types, and any risk-specific notes — everything else is here.

When in doubt, fall back to report-only.

## Localized fix bar

A maintenance task may be converted to a PR only if all hold:

- **Bounded scope**: ≤3 files, ≤50 changed lines (added + removed). For
  docs-and-references, allow ≤10 files and ≤200 changed lines. For dependencies,
  exclude generated `uv.lock` changes from the LOC total; the lockfile still
  counts toward the file cap and must pass the path allowlist.
- **Reversible**: no public API changes, no `__all__` deletions, no version
  bumps (Dependabot owns those), no schema changes, no migrations.
- **Self-evident**: the audit established both the problem *and* the unique
  correct fix. Mechanical, not interpretive.
- **Test-safe**: when the recipe declares `test_required`, run the
  per-package test target for the affected package and abort on failure.
  Mapping (the Makefile does not expose `test-<package>` directly):

  | Package directory | Test target |
  |-------------------|-------------|
  | `packages/data-designer-config` | `make test-config` |
  | `packages/data-designer-engine` | `make test-engine` |
  | `packages/data-designer` | `make test-interface` |
- **Complete concern**: finish one bounded maintenance task. Include related
  eligible findings rather than leaving known mechanical follow-ups for later
  PRs. Suite-declared batches share a category and test target; docs tasks may
  span eligible categories within one subsystem or API migration. The combined
  diff must satisfy the scope limits. If a task cannot fit, report it for
  maintainer action instead of splitting it into a sequence of tiny PRs.
- **Worth reviewing (docs-and-references)**: fix a concrete runtime/install
  failure, broken public example, or broken user-facing documentation link;
  otherwise require at least two related eligible findings. An isolated
  cosmetic or metadata-hygiene finding stays in `fix_backlog`, even if it is
  easy or old. Do not bundle
  unrelated findings or add changes just to meet this bar. Deferral is not a
  failed attempt and must not add an `attempted_fixes` entry.
  Other suites retain their recipe-defined singleton eligibility and batching
  rules; this threshold must not disable non-batchable code-fix categories.
- **Allowlisted paths**: matches the suite's path allowlist.

If the top-ranked candidate fails the bar, try the next. If none of the top
5 qualify, skip the fix step and emit report-only.

## Allowlists

### Per-suite path allowlist

| Suite | Paths the recipe MAY modify |
|-------|-----------------------------|
| docs-and-references | `architecture/**`, `fern/versions/latest/pages/**` (excluding `devnotes/**`), `README.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, `STYLEGUIDE.md`, `packages/*/src/**/*.py` (docstring-only edits) |
| dependencies | `packages/*/pyproject.toml`, `uv.lock` |
| structure | `packages/*/src/**/*.py` |
| code-quality | `packages/*/src/**/*.py` |
| test-health | (no fix phase) |

### Shared forbidden paths (all suites)

- `.github/workflows/**`, `.agents/**`, repo-root `pyproject.toml`,
  `.git/**`, anything in `.gitignore`.

### Shared forbidden commands

- `git push --force` (any variant), `git rebase`, `git reset --hard`,
  `git branch -D`/`-d`/`--delete`.
- `gh pr merge`, `gh pr close`, `gh pr review`.
- `pip install`, `uv pip install` (use `make install-dev` only).

## Runner-state schema

Each daily recipe maintains two arrays in
`{{memory_path}}/runner-state.json` beyond the existing `known_issues` /
`baselines`:

```json
{
  "fix_backlog": [
    { "id": "<hash>", "category": "...", "first_seen": "YYYY-MM-DD",
      "last_seen": "YYYY-MM-DD", "data": { /* category fields */ } }
  ],
  "attempted_fixes": [
    { "id": "<hash>", "attempts": [
      { "pr_number": 612, "outcome": "merged", "at": "YYYY-MM-DD",
        "branch": "agentic-ci/..." }
    ] }
  ]
}
```

Also: `draft_until_proven` (boolean, per-suite, default `true` for
code-quality and unset elsewhere) controls draft-PR mode.

Batch PRs still record one `attempted_fixes` entry per finding. Multiple
entries may point to the same `pr_number` and `branch`.
Post-fix gates group those entries by `(pr_number, branch)`, validate the
shared diff once, and update every associated finding together on rejection.

### `fix_backlog` rules (audit phase populates this)

- Append every detected finding in an eligible category. If `id` is already
  present, **refresh `last_seen` and merge the current scan's values into
  `data`**, preserving `blocked_reason` and `blocked_at` as described below.
  The `data` field is used by the fix phase to apply the change
  without re-scanning, so stale `data` would let an old plan drive a new
  PR after the underlying file moved or changed.
- Drop entries with `last_seen` older than 30 days.
- Cap at 200 entries (drop oldest by `first_seen`).
- Populated **before** the `known_issues` filter so fixable findings persist
  even when their report row is suppressed for being unchanged.
- Batchable categories must include enough information in `data` to group
  siblings safely. For package-scoped Python fixes, derive `test_target` from
  the package containing the source file.
- Preserve `data.blocked_reason` and `data.blocked_at` across audit refreshes.
  Clear them only when new source evidence or maintainer action resolves the
  recorded blocker, not merely because the finding was seen again.
- Docs findings also carry `data.task_key` (a stable subsystem or API-migration
  key) and `data.scope_complete` (whether that task's bounded audit finished).
  Reset `scope_complete` to `false` on existing docs backlog entries at the
  start of each audit. Set it to `true` for the selected task only after all
  related references in its declared scope have been checked. Cached entries
  without these fields are not fixable until refreshed by an audit.

### `attempted_fixes` rules

`outcome` ∈ `{open, merged, closed, abandoned}`.

- `abandoned` means the recipe could not produce a PR (tests failed,
  conflict, lint failed, allowlist rejected, etc.).
- Reconcile at the start of each fix run. First refresh existing latest
  `open` attempts that have a `pr_number`: query the PR and flip the
  attempt to `merged` or `closed` if it is no longer open. Then recover
  from crashes that left state un-updated: list open PRs (`gh pr list`)
  whose bodies contain one or more
  `<!-- agentic-ci finding=<id> suite=<suite> -->` markers, parse out
  every `<id>`, and back-fill any missing `attempted_fixes` entries with
  `outcome: "open"` and the parsed `pr_number` and `branch`.
- Prune: drop `merged` entries older than 90 days. Do **not** prune
  `closed` or `abandoned` entries by age — pruning a single-strike entry
  would erase the history needed to ever reach the two-strike threshold.
- The 200-entry cap handles long-tail cleanup. Eviction order:
  non-two-strike entries first, oldest-first by `attempts[0].at`.
  Two-strike entries (≥2 `closed`/`abandoned`) are exempt from cap
  eviction unless every other entry has already been evicted — they
  represent maintainer-action signals and must not be silently
  forgotten. If two-strike entries alone exceed 200, that's itself a
  signal worth surfacing; in that pathological case, evict oldest-first
  by `attempts[0].at`.
- Two-strike entries surface in the report under
  `Repeatedly-failed fix attempts` and are filtered from selection
  permanently.

## Finding hash

`finding_id = sha1(suite + ":" + canonical_key)[:12]`, where
`canonical_key` uses durable identifiers only — never line numbers or free
text:

| Suite (category) | canonical_key |
|------------------|---------------|
| docs (broken-link) | `<source-file>:<target>` |
| docs (docstring-drift) | `<source-file>:<symbol>:<param-or-empty>:<drift-type>` |
| docs (arch-ref-rename) | `<doc-file>:<old-symbol>` |
| dependencies (transitive-gap) | `<package>:<dep>:transitive` |
| dependencies (unused) | `<package>:<dep>:unused` |
| structure (missing-future) | `<source-file>:missing-future` |
| structure (lazy-import) | `<source-file>:lazy-import:<imported-module>` |
| code-quality (bare-except) | `<source-file>:<enclosing-symbol>:<try-body-hash>:<ordinal>:bare-except` |

Symbols use fully-qualified Python names.
`try-body-hash` is `sha1(<try-block body, leading/trailing whitespace
stripped, internal lines preserved>)[:8]`.
`ordinal` is the 1-based position of this bare-except among bare-excepts
in the same enclosing symbol, in source order. Both are needed: the body
hash distinguishes most cases, and the ordinal disambiguates the rare
case of two bare-except blocks with byte-identical try bodies.

## Ranking

Confidence is an eligibility gate: every included finding must meet its
category's evidence requirements and have one unambiguous fix. Among eligible
tasks, earlier criteria override later ones:

1. **Defect severity**:

   | Severity | Examples |
   |----------|----------|
   | high | imported dependency not guaranteed by any required package, heavy import bypassing lazy system |
   | medium | broken doc link visible on docs site, bare-except hiding errors, docstring drift on public API |
   | low | direct-dependency gap already guaranteed by a required package, stale internal class reference, missing `__future__ import annotations`, unused dep |

2. **User-facing impact** - visible to docs-site readers or plugin
   consumers vs internal-only.

3. **Task completeness** - prefer resolving a related group of findings over
   an isolated edit. Finding count alone is not impact.

4. **Recency** - newer findings rank above long-standing ones.

Record the chosen finding id(s), impact evidence, and grouping rationale at the top of
`/tmp/audit-{{suite}}.md`.

## Standard fix procedure

The fix phase of every eligible recipe follows these steps. Suite recipes
declare only the parts that vary (eligible categories, branch type,
`test_required`, suite-specific quirks).

1. Reconcile `attempted_fixes`: refresh recorded open PRs to
   `merged`/`closed` when appropriate, then scan open PRs (`gh pr list`)
   to recover any state lost to a prior crash.
2. Filter `fix_backlog`: drop entries whose latest attempt is `open` or
   `merged`; surface two-strike entries in the report's
   `Repeatedly-failed fix attempts` section and drop them from selection.
3. Group the remainder into tasks per the suite recipe, then rank per the
   Ranking section. Docs tasks require `data.scope_complete: true` on every
   entry. Skip tasks with any member carrying `data.blocked_reason` and surface
   the blocker for maintainer action. Skip tasks below the worth-reviewing bar without recording attempts;
   leave them in the backlog and explain the deferral in the report.
4. For each primary candidate, top 5 max:
   1. If the suite declares the category batchable, collect sibling
      `fix_backlog` entries for the same suite/category that share the same
      target package, test target, and branch type. Docs instead collect all
      eligible entries with the same `data.task_key`, across eligible
      categories. Do not discover new findings; use only existing backlog
      entries. Code batches remain capped at 3 entries; docs tasks use the
      docs scope limits above.
   2. Re-verify every finding still applies (re-grep / re-read). If a
      sibling no longer applies, remove it from `fix_backlog`; if the
      primary no longer applies, remove it from `fix_backlog` and continue
      to the next primary candidate.
   3. Recheck the worth-reviewing bar after removing stale findings. Defer a
      task that no longer meets it. If a known member cannot be fixed safely
      during re-verification, defer the whole
      task for maintainer action without recording attempts for any member.
      Persist `data.blocked_reason` and `data.blocked_at` on each task member
      so later runs skip the task until an audit resolves the blocker.
      Do not omit that member to produce a partial PR. Only failures after
      beginning edits or PR creation count as abandoned attempts.
      Apply the complete task only after every member passes re-verification.
      If the combined diff exceeds the localized-fix bar or touches a
      non-allowlisted path, abandon and continue.
   4. If the category sets `test_required: true`, run the per-package
      test target (see the mapping table in "Localized fix bar" above)
      for every affected package, once per target. For mixed docs tasks,
      tests are required if any member requires them. On failure: abandon
      and continue.
   5. Branch: `agentic-ci/<type>/<suite>-YYYYMMDD-<short-slug>`. Commit:
      `<type>(agentic-ci): <one-line>`. Push.
   6. Write the PR body to `/tmp/pr-body-{{suite}}.md`, including one
      hidden metadata block per fixed finding:
      `<!-- agentic-ci finding=<id> suite=<suite> -->`
   7. `gh pr create --body-file /tmp/pr-body-{{suite}}.md` with `--draft`
      iff `draft_until_proven` is true for the suite.
   8. `gh pr edit <num> --add-label agentic-ci --add-label agentic-ci/<suite>`.
   9. Record one `attempted_fixes` entry per fixed finding with
      `outcome: "open"` and exit.
5. If no task qualifies, append a short note to the report and exit cleanly.
   Distinguish deferred tasks from abandoned attempts.

On any failure mid-flow: record `outcome: "abandoned"` for the chosen
finding(s) (with `pr_number: null`), leave any pushed branch in place
(`pr-stale.yml` will reap it; branch deletion is forbidden), and continue
to the next candidate.

## PR conventions

- **Use `gh pr create --body-file`**, not `/create-pr`. The skill is
  interactive-only and shells the body inline; CI needs determinism.
- **Title**: conventional, `<type>(agentic-ci): <one-line>`.
- **Labels**: `agentic-ci`, `agentic-ci/<suite>`.
- **Batch markers**: batch PRs include one hidden finding marker per fixed
  finding so crash recovery can reconstruct every `attempted_fixes` entry.
- **Task summary**: describe the completed maintenance task, its impact, and
  every included finding. Explain any related report-only findings rather
  than silently leaving mechanical follow-ups out of scope.
- **Draft PRs**: `code-quality` opens draft until a maintainer flips
  `draft_until_proven` to `false` in runner-state, after at least two
  non-draft PRs from that suite have landed clean. This flip is
  intentionally manual — it is the sole human-gated promotion step in
  the fix policy and must not be automated.

## Atomicity

Each fix-phase invocation produces exactly one of:

- **Report-only** — runner-state updated; no branch, commit, or PR.
- **Report + PR** — same, plus a pushed branch, a commit, and a PR. The
  `attempted_fixes` entry is recorded *before* the recipe exits.

No half-states. The runner state is the source of truth for what the
recipe has tried; never silently drop a failed attempt.

The matrix-level concurrency for the daily workflow uses
`cancel-in-progress: false` so a fix in flight cannot be cancelled
between push and PR open. The trade-off is a queued duplicate run if a
manual dispatch arrives while cron is still going; that's preferable to
orphaned branches with no `attempted_fixes` record.

## Workflow-level scope gate

The agent's compliance with the path allowlists and the localized-fix
bar is load-bearing for autonomous PR generation, but the recipe alone
cannot enforce them. The daily workflow runs a post-fix scope gate that
re-derives the per-suite allowlist (mirrored from the table above) and
the diff stats from the pushed branch, then closes the PR and deletes
the remote branch on violation. The gate also flips the
`attempted_fixes` entry from `open` to `abandoned` so two-strike logic
sees the failure. Keep the workflow's allowlist regexes in sync with the
table above; the workflow is the enforcement, the table is the
specification.
