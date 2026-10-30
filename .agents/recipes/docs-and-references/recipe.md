---
name: docs-and-references
description: Audit documentation freshness - docstrings vs signatures, broken links, architecture refs, docs site content accuracy
trigger: schedule
tool: claude-code
timeout_minutes: 40
max_turns: 50
permissions:
  contents: write
---

# Documentation and References Audit

Check that documentation stays in sync with code. Write findings to
`/tmp/audit-{{suite}}.md`.

This repo has no ruff D* docstring rules enabled, so docstring quality is
not enforced by CI. This recipe fills that gap with cross-referencing that
a linter can't do: checking docstrings against actual signatures, docs
against actual code, and links against actual targets.

## Runner memory

Read `{{memory_path}}/runner-state.json` for known issues from previous runs.
After completing the audit, update the file with any new findings (add to
`known_issues` array with a short hash of the finding). Skip reporting issues
that already appear in `known_issues`.

This recipe also maintains `fix_backlog` and `attempted_fixes` per
`_fix-policy.md`. Update `fix_backlog` for every detected finding *before*
the `known_issues` filter applies, so fixable findings persist across runs
even when their report row is suppressed for being unchanged.

## Instructions

### Turn budget

This suite must finish before the `max_turns` limit. Do not attempt a
repo-wide audit in one run. Complete one maintenance task instead of
sampling unrelated sections until the first easy fix appears.

1. Read runner memory.
2. Write `/tmp/audit-{{suite}}.md` immediately with the required headings and
   empty tables. If the run is interrupted later, the workflow must still have
   a usable partial report. Reset `data.scope_complete` to `false` on all
   existing docs backlog entries before scanning.
3. Choose one subsystem or API migration using pending backlog entries and
   recent code changes. Prefer user-facing breakage, then related accumulated
   drift. The workflow reconciles recorded open PRs before this audit so
   closed PRs can be retried without another task triggering the fix phase.
   Exclude findings whose latest attempt is open or merged, or that
   have two closed/abandoned attempts. Rotate away from `docs_last_task_key`
   when another eligible scope is available, especially if the previous scope
   was incomplete or deferred. Keep oversized or unsafe scopes report-only
   until their per-entry `data.blocked_reason` is resolved by new evidence
   or maintainer action. Declare its stable `task_key`, relevant
   symbols/settings, and paths in the report (for example,
   `models/request-admission`).
4. Search the allowlisted documentation for those symbols, renamed settings,
   links, and examples. Inspect the related matches as one scope, up to 10
   editable files, plus the source needed to establish the correct behavior.
   Apply the checks below within that scope. Collect every related eligible
   finding, including already-known findings; do not stop after two findings
   or after finding one rename. Refresh each entry's `data.task_key`, proposed
   fix, impact evidence, and `last_seen`. Remove entries in the selected scope
   that no longer apply.
5. Use at most 20 tool calls, reserving the last 3 for the report and state.
   If the scope cannot be fully checked within this budget or the docs diff
   limits, leave it incomplete and record what remains in `data.blocked_reason`
   and `data.blocked_at` on its backlog entries. Do not mark a sampled
   subset complete to produce a smaller PR. Otherwise set
   `data.scope_complete: true` on every eligible entry in the task.
6. Finalize the report and runner memory. Save `docs_last_task_key`.
   Record scope completion, expected impact, and any deferred
   findings. Write `NO_FINDINGS` only if no findings
   were verified, including previously known findings.

### 1. Docstring vs signature drift

This repo uses Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
Check public functions and methods in the selected scope for mismatches
between the docstring and the actual function signature:

- Parameters in the `Args:` section that no longer exist in the signature
- Parameters in the signature that are missing from `Args:`
- `Returns:` section that contradicts the return type annotation
- `Raises:` section listing exceptions the function can no longer raise

Focus on public API surface: `__init__`, public methods (no leading
underscore), and module-level functions in `packages/*/src/`. Skip test
files, private methods, and `__dunder__` methods other than `__init__`.

Prefer public interface entry points when selecting a task. Follow related
engine and config symbols as needed to finish that task.

### 2. Broken internal links

Check links related to the selected scope in these locations:
- `README.md` - relative links
- `architecture/*.md` - cross-references to other architecture docs and code
- `fern/versions/latest/pages/` - Fern content links, code references, cross-page links
- `CONTRIBUTING.md`, `DEVELOPMENT.md`, `STYLEGUIDE.md` - relative links

Use targeted link extraction. For each related internal link, verify the
target file or anchor exists. Report broken links with the source file, line
number, and broken target.

### 3. Architecture doc references

The 10 files in `architecture/` reference specific classes, functions, files,
and registries by name. These are high-value docs that agents and developers
rely on for orientation. Inspect the architecture files in the selected
scope. For each related code or configuration reference:
- Verify the referenced class, function, or module still exists at the stated
  location
- If renamed or moved, flag with the old and new location
- Check adjacent settings and examples for the same migration, including
  deprecated aliases whose canonical replacements are explicit in source

```bash
ls architecture/
# Key files: overview.md, config.md, engine.md, dataset-builders.md,
# models.md, sampling.md, cli.md, plugins.md, mcp.md, agent-introspection.md
```

### 4. Docs site content accuracy

The Fern site under `fern/versions/latest/pages/` is the primary user-facing
documentation. Check pages in the selected scope against the current code:

**Concepts pages** (`fern/versions/latest/pages/concepts/`):
- Do code examples use correct imports, class names, and method signatures?
  Check against actual source - e.g., verify `DataDesigner.create()`,
  `DataDesigner.preview()`, builder patterns match the real API.
- Are there documented config options or column types that have been removed
  or renamed?
- Are new features or column types missing from the docs?

**Recipes** (`fern/versions/latest/pages/recipes/`):
- Do step-by-step instructions reference correct file paths, class names,
  and CLI commands? Run `grep` for class names mentioned in recipe docs and
  verify they resolve in the source.

**Dev notes** (`fern/versions/latest/pages/devnotes/posts/`):
- Dev notes describe implementation details that may have changed. Flag
  references in the selected scope that need a maintainer's judgement.

Keep unrelated pages outside the selected task. Dev-note rewrites and
semantic changes still stay report-only.

## Output format

Write the report to `/tmp/audit-{{suite}}.md`:

```markdown
<!-- agentic-ci-daily-{{suite}} -->
## Documentation Audit - {{date}}

**Workflow run:** see GitHub Actions

### Maintenance task

- Task key: `<subsystem-or-api-migration>`
- Scope: paths and symbols checked
- Audit complete: yes/no (remaining work if no)
- Impact and disposition: proposed task, deferred cosmetic singleton, or report-only

### Docstring vs signature drift

| File | Function | Issue |
|------|----------|-------|
| ... | ... | Param `x` removed from signature but still in Args |

### Broken links

| Source file | Line | Target | Status |
|-------------|------|--------|--------|
| ... | ... | ... | 404 / anchor missing |

### Stale architecture references

| Doc | Reference | Issue |
|-----|-----------|-------|
| ... | `FooClass` | Renamed to `BarClass` in engine/... |

### Docs site accuracy

| Page | Issue | Severity |
|------|-------|----------|
| ... | `DataDesigner.foo()` removed in v0.3 | high - user-facing |

### Summary

- N docstring mismatches (M new since last run)
- N broken links (M new)
- N stale architecture refs (M new)
- N docs accuracy issues (M new)
```

If no findings in any category, write `NO_FINDINGS` on the first line instead.

## Fix phase

Follow the standard fix procedure in `_fix-policy.md`. Suite-specific bits:

### Eligible categories

| Category | Branch type | test_required | Eligibility note |
|----------|-------------|---------------|------------------|
| broken-link | `docs` | no | Only when the corrected target is unambiguous (exact-match file at a different path, or a single similar anchor). Multiple candidates → ineligible. |
| docstring-drift | `docs` | yes | Purely signature-driven `Args:`/`Returns:`/`Raises:` updates. Rename a param to its current name, drop entries for removed params, add placeholder entries for added params (note the signature; do not invent semantic descriptions). |
| arch-ref-rename | `docs` | no | Mechanical code/config reference updates in architecture or user docs, excluding dev notes. Source must establish one replacement through a rename, move, or explicit deprecated-alias mapping. Check related example imports and settings against source; do not infer new behavior. |

`fix_backlog.data` should carry whatever the fix step needs without
re-scanning: the proposed target for broken-link, the signature-vs-Args
delta for docstring-drift, the new symbol or canonical setting name and
source evidence for arch-ref-rename. Every entry also needs `task_key` and
`scope_complete` per `_fix-policy.md`.

All three categories may be grouped in one docs task with the same `task_key`,
up to 10 files and 200 changed lines. Include all related eligible findings
from the completed audit, across architecture docs, user docs, and docstrings.
Run each required package test target once if any task member needs tests.
Re-verify every member, but do not discover more findings in the fix phase.
Defer incomplete scopes and isolated cosmetic fixes without recording failed
attempts. A broken public example or user-facing link can justify a singleton.

For example, a model-docs task should update the removed controller name and
its explicitly deprecated ramp-setting alias together, along with related
links or examples found in that audit. Do not leave a known mechanical
correction in the same task out of scope for a future one-line PR.

All other audit categories (docs-site rewrites, dev-note edits, external
URL breakage) stay report-only.

## Constraints

- Outside the fix phase, this recipe is read-only — do not modify files.
- Within the fix phase, only modify paths in the suite's path allowlist.
  See `_fix-policy.md` for the shared command/path baseline.
- Do not read file contents unless needed to verify a specific reference.
  Use `grep` and `head` for targeted checks rather than reading entire files.
- Skip vendored or generated files.
- License headers are already enforced by the `license-headers` CI job.
  Do not check for SPDX headers.
- Ruff lint and format are already enforced by CI. Do not duplicate those
  checks. Focus on cross-references that require understanding both the docs
  and the code.
