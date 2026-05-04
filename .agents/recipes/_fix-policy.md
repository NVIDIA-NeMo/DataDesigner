# Agentic CI Fix Policy

Prepended to every daily-suite recipe alongside `_runner.md`. Defines what
"open a PR" means for these recipes and the rules that apply across all of
them. Each suite recipe declares only its eligible finding categories, its
branch types, and any risk-specific notes — everything else is here.

When in doubt, fall back to report-only.

## Localized fix bar

A finding may be converted to a fix only if all hold:

- **Bounded scope**: ≤3 files, ≤50 LOC net.
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
- **Single concern**: one finding per PR.
- **Allowlisted paths**: matches the suite's path allowlist.

If the top-ranked candidate fails the bar, try the next. If none of the top
5 qualify, skip the fix step and emit report-only.

## Allowlists

### Per-suite path allowlist

| Suite | Paths the recipe MAY modify |
|-------|-----------------------------|
| docs-and-references | `architecture/**`, `docs/**`, `README.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, `STYLEGUIDE.md`, `packages/*/src/**/*.py` (docstring-only edits) |
| dependencies | `packages/*/pyproject.toml` |
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

### `fix_backlog` rules (audit phase populates this)

- Append every detected finding in an eligible category. If `id` is already
  present, **refresh both `last_seen` and `data`** with the current scan's
  values. The `data` field is used by the fix phase to apply the change
  without re-scanning, so stale `data` would let an old plan drive a new
  PR after the underlying file moved or changed.
- Drop entries with `last_seen` older than 30 days.
- Cap at 200 entries (drop oldest by `first_seen`).
- Populated **before** the `known_issues` filter so fixable findings persist
  even when their report row is suppressed for being unchanged.

### `attempted_fixes` rules

`outcome` ∈ `{open, merged, closed, abandoned}`.

- `abandoned` means the recipe could not produce a PR (tests failed,
  conflict, lint failed, allowlist rejected, etc.).
- Reconcile against open PRs (`gh pr list`) at the start of each fix run
  to recover from crashes that left state un-updated.
- Prune: drop `merged` entries older than 90 days. Do **not** prune
  `closed` or `abandoned` entries by age — pruning a single-strike entry
  would erase the history needed to ever reach the two-strike threshold.
  The 200-entry cap (with oldest-first eviction by `first_seen`) handles
  long-tail cleanup.
- Two-strike entries (≥2 `closed`/`abandoned`) surface in the report
  under `Repeatedly-failed fix attempts` and are filtered from selection
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
| code-quality (bare-except) | `<source-file>:<enclosing-symbol>:<try-body-hash>:bare-except` |

Symbols use fully-qualified Python names.
`try-body-hash` is `sha1(<try-block body, leading/trailing whitespace
stripped, internal lines preserved>)[:8]` — needed because a function
can contain multiple `try:` blocks with bare excepts that would
otherwise collide on the same finding id.

## Ranking

Earlier criteria override later ones:

1. **Fix confidence** (per-category):

   | Category | Confidence |
   |----------|-----------|
   | structure / missing-future | 1.0 |
   | structure / lazy-import | 0.9 |
   | docs / broken-link | 0.9 |
   | dependencies / transitive-gap | 0.85 |
   | docs / arch-ref-rename | 0.8 |
   | dependencies / unused | 0.75 |
   | docs / docstring-drift | 0.75 |
   | code-quality / bare-except | 0.6 |

2. **Defect severity**:

   | Severity | Examples |
   |----------|----------|
   | high | missing transitive dep, heavy import bypassing lazy system |
   | medium | broken doc link visible on docs site, bare-except hiding errors, docstring drift on public API |
   | low | broken link in dev-notes, missing `__future__ import annotations`, unused dep |

3. **User-facing impact** — visible to docs-site readers or plugin
   consumers vs internal-only.

4. **Recency** — newer findings rank above long-standing ones.

Record the chosen finding's id, scores, and rationale at the top of
`/tmp/audit-{{suite}}.md`.

## Standard fix procedure

The fix phase of every eligible recipe follows these steps. Suite recipes
declare only the parts that vary (eligible categories, branch type,
`test_required`, suite-specific quirks).

1. Reconcile `attempted_fixes` against open PRs (`gh pr list`) to recover
   any state lost to a prior crash.
2. Filter `fix_backlog`: drop entries whose latest attempt is `open` or
   `merged`; surface two-strike entries in the report's
   `Repeatedly-failed fix attempts` section and drop them from selection.
3. Rank the remainder per the Ranking section.
4. For each candidate, top 5 max:
   1. Re-verify the finding still applies (re-grep / re-read). If not,
      remove from `fix_backlog` and continue.
   2. Apply the fix. If the diff exceeds the localized-fix bar or touches
      a non-allowlisted path, abandon and continue.
   3. If the category sets `test_required: true`, run
      `make test-<package>` for the package containing the change. On
      failure: abandon and continue.
   4. Branch: `agentic-ci/<type>/<suite>-YYYYMMDD-<short-slug>`. Commit:
      `<type>(agentic-ci): <one-line>`. Push.
   5. Write the PR body to `/tmp/pr-body-{{suite}}.md`, including the
      hidden metadata block:
      `<!-- agentic-ci finding=<id> suite=<suite> -->`
   6. `gh pr create --body-file /tmp/pr-body-{{suite}}.md` with `--draft`
      iff `draft_until_proven` is true for the suite.
   7. `gh pr edit <num> --add-label agentic-ci --add-label agentic-ci/<suite>`.
   8. Record `attempted_fixes` entry with `outcome: "open"` and exit.
5. If all 5 candidates were abandoned, append a one-line note to the
   report and exit cleanly. The state already reflects the abandonments.

On any failure mid-flow: record `outcome: "abandoned"` for the chosen
finding (with `pr_number: null`), leave any pushed branch in place
(`pr-stale.yml` will reap it; branch deletion is forbidden), and continue
to the next candidate.

## PR conventions

- **Use `gh pr create --body-file`**, not `/create-pr`. The skill is
  interactive-only and shells the body inline; CI needs determinism.
- **Title**: conventional, `<type>(agentic-ci): <one-line>`.
- **Labels**: `agentic-ci`, `agentic-ci/<suite>`.
- **Draft PRs**: `code-quality` opens draft until a maintainer flips
  `draft_until_proven` to `false` in runner-state, after at least two
  non-draft PRs from that suite have landed clean.

## Atomicity

Each fix-phase invocation produces exactly one of:

- **Report-only** — runner-state updated; no branch, commit, or PR.
- **Report + PR** — same, plus a pushed branch, a commit, and a PR. The
  `attempted_fixes` entry is recorded *before* the recipe exits.

No half-states. The runner state is the source of truth for what the
recipe has tried; never silently drop a failed attempt.
