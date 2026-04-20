---
name: docs-and-references
description: Audit documentation freshness - docstrings vs signatures, broken links, architecture refs, docs site content accuracy
trigger: schedule
tool: claude-code
timeout_minutes: 20
max_turns: 30
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

## Instructions

### 1. Docstring vs signature drift

This repo uses Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
Scan public functions and methods in `packages/` for mismatches between the
docstring and the actual function signature:

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

For each link, verify the target file or anchor exists. Report broken links
with the source file, line number, and broken target.

### 3. Architecture doc references

The 10 files in `architecture/` reference specific classes, functions, files,
and registries by name. These are high-value docs that agents and developers
rely on for orientation. For each code reference:
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

**Code reference** (`docs/code_reference/`):
- Check that autodoc module paths point to modules that still exist.

**Prioritize by risk of drift**: pages with the most code symbols referenced
are most likely to be stale. Don't read every page - sample 5-10 high-value
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

## Constraints

- Do not modify any files. This is a read-only audit.
- Do not read file contents unless needed to verify a specific reference.
  Use `grep` and `head` for targeted checks rather than reading entire files.
- Skip vendored or generated files.
- License headers are already enforced by the `license-headers` CI job.
  Do not check for SPDX headers.
- Ruff lint and format are already enforced by CI. Do not duplicate those
  checks. Focus on cross-references that require understanding both the docs
  and the code.
