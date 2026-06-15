---
name: docs-and-references
description: Audit documentation freshness - docstrings vs signatures, broken links, architecture refs, docs site content accuracy
trigger: schedule
tool: claude-code
timeout_minutes: 20
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
repo-wide audit in one run.

1. Read runner memory.
2. Write `/tmp/audit-{{suite}}.md` immediately with the required headings and
   empty tables. If the run is interrupted later, the workflow must still have
   a usable partial report.
3. Use targeted searches to find candidates, then read only the files needed
   to verify a specific finding.
4. Stop after either:
   - 20 tool calls
   - 2 new findings in a section
   - all sections have been sampled
5. Finalize the report, update runner memory, and stop. If no new findings
   were verified, replace the report with `NO_FINDINGS`.

### 1. Docstring vs signature drift

This repo uses Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
Sample public functions and methods in `packages/` for mismatches between the
docstring and the actual function signature. Do not scan every source file.
Use `rg "Args:|Returns:|Raises:" packages/*/src/ --glob '*.py'` to find
candidates, then inspect at most 5 high-value files:

- Parameters in the `Args:` section that no longer exist in the signature
- Parameters in the signature that are missing from `Args:`
- `Returns:` section that contradicts the return type annotation
- `Raises:` section listing exceptions the function can no longer raise

Focus on public API surface: `__init__`, public methods (no leading
underscore), and module-level functions in `packages/*/src/`. Skip test
files, private methods, and `__dunder__` methods other than `__init__`.

**Prioritize by impact**: start with `packages/data-designer/src/` (public
interface), then `packages/data-designer-engine/src/`, then config. The
interface package is what users see first.

### 2. Broken internal links

Check links in these locations:
- `README.md` - all relative links and URLs
- `architecture/*.md` - cross-references to other architecture docs and code
- `docs/` - MkDocs content links, code references, cross-page links
- `CONTRIBUTING.md`, `DEVELOPMENT.md`, `STYLEGUIDE.md` - relative links

Use targeted link extraction and inspect at most 10 candidate links. Prefer
high-value docs and links changed recently. For each sampled link, verify the
target file or anchor exists. Report broken links with the source file, line
number, and broken target.

### 3. Architecture doc references

The 10 files in `architecture/` reference specific classes, functions, files,
and registries by name. These are high-value docs that agents and developers
rely on for orientation. Sample at most 3 architecture files per run,
prioritizing files changed recently. For each code reference:
- Verify the referenced class, function, or module still exists at the stated
  location
- If renamed or moved, flag with the old and new location

```bash
ls architecture/
# Key files: overview.md, config.md, engine.md, dataset-builders.md,
# models.md, sampling.md, cli.md, plugins.md, mcp.md, agent-introspection.md
```

### 4. Docs site content accuracy

The MkDocs site under `docs/` is the primary user-facing documentation.
Review for accuracy against the current code:

**Concepts pages** (`docs/concepts/`):
- Do code examples use correct imports, class names, and method signatures?
  Check against actual source - e.g., verify `DataDesigner.create()`,
  `DataDesigner.preview()`, builder patterns match the real API.
- Are there documented config options or column types that have been removed
  or renamed?
- Are new features or column types missing from the docs?

**Recipes** (`docs/recipes/`):
- Do step-by-step instructions reference correct file paths, class names,
  and CLI commands? Run `grep` for class names mentioned in recipe docs and
  verify they resolve in the source.

**Dev notes** (`docs/devnotes/posts/`):
- Dev notes describe implementation details that may have changed. Spot-check
  the most recent 3-5 posts for references to functions, classes, or
  architecture that have since been modified.

**Prioritize by risk of drift**: pages with the most code symbols referenced
are most likely to be stale. Don't read every page - sample 3-5 high-value
pages and flag patterns.

## Output format

Write the report to `/tmp/audit-{{suite}}.md`:

```markdown
<!-- agentic-ci-daily-{{suite}} -->
## Documentation Audit - {{date}}

**Workflow run:** see GitHub Actions

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
| arch-ref-rename | `docs` | no | Only when grep confirms the old symbol is gone and exactly one similarly-named new symbol exists at the same role. |

`fix_backlog.data` should carry whatever the fix step needs without
re-scanning: the proposed target for broken-link, the signature-vs-Args
delta for docstring-drift, the new symbol name for arch-ref-rename.

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
