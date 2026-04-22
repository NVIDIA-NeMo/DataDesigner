---
name: dependencies
description: Audit dependency health - version pinning, transitive gaps, CVEs, unused deps, cross-package consistency
trigger: schedule
tool: claude-code
timeout_minutes: 20
max_turns: 30
permissions:
  contents: write
---

# Dependency Audit

Audit the dependency graph across all three packages. Write findings to
`/tmp/audit-{{suite}}.md`.

Dependabot handles version bump PRs. This recipe focuses on what Dependabot
can't do: cross-package consistency, transitive dependency gaps, unused deps,
and pinning strategy review that requires understanding the library's
stability history.

## Runner memory

Read `{{memory_path}}/runner-state.json` for known issues and previously seen
dependency versions. After the audit, update `known_issues` and
`baselines.dependency_versions` with the current state. Skip reporting issues
that already appear in `known_issues`.

## Instructions

### 1. Inventory current dependencies

Read the `pyproject.toml` for each package:
```bash
cat packages/data-designer-config/pyproject.toml
cat packages/data-designer-engine/pyproject.toml
cat packages/data-designer/pyproject.toml
```

Note: engine and interface packages use `uv-dynamic-versioning` to inject
dependencies. Check both static declarations and the dynamic versioning
config.

### 2. Transitive dependency gaps

This is the highest-value check. A package may import a library that it
doesn't declare as a dependency, relying on another package to pull it in
transitively. This works until the packages are installed separately.

For each package, verify that every imported third-party library is declared
in that package's own `[project.dependencies]`:

```bash
# Find all third-party imports in engine source
grep -rhn "^import \|^from " packages/data-designer-engine/src/ --include='*.py' \
  | grep -v "data_designer" | grep -v "^from __future__" \
  | sort -u

# Compare against declared deps in engine's pyproject.toml
```

Known issue to check: `numpy` and `pandas` are used by the engine but may
only be declared in the config package. Each package should declare what it
directly imports.

Also check lazy imports in `data_designer/lazy_heavy_imports.py` - these
are intentionally deferred but still need to be declared as dependencies.

### 3. Cross-package version consistency

Check that shared dependencies use consistent version constraints:
```bash
# Extract dependency specs from all three pyproject.toml files
grep -E "^\s+\"[a-zA-Z]" packages/*/pyproject.toml
```

Flag cases where the same package has conflicting version ranges across
packages (e.g., `pandas>=2.0` in config but `pandas>=1.5` in engine).

Also check the CVE minimum version constraints already established in the
repo (look for `[tool.uv.constraint-dependencies]` sections) and verify
they haven't been bypassed by a looser pin elsewhere.

### 4. Unused dependency detection

For each declared dependency, check if it is actually imported anywhere in
the corresponding package:
```bash
# Example: check if 'lxml' is imported in data-designer-engine
grep -r "import lxml\|from lxml" packages/data-designer-engine/src/
```

A dependency is "unused" if:
- Not imported directly anywhere in the package source
- Not used via the lazy import system (`lazy_heavy_imports`)
- Not a plugin entry point or runtime-only requirement
- Not a build/test-only dependency incorrectly in `[project.dependencies]`

### 5. Version pinning review

For each dependency, assess the pinning strategy. The repo currently uses
a mix of bounded ranges (`>=X,<Y`) and loose pins (`>=X`).

Flag only high-risk cases:
- **Unbounded pins on libraries with breaking-change history**: litellm,
  pydantic, and similar libraries that have broken APIs between minor
  versions should use strict or compatible-release pins
- **Overly strict pins that block security updates**: if a dependency has
  a CVE fix in a newer minor version but the pin prevents upgrading

Do NOT recommend blanket strict-pinning. This repo intentionally uses
bounded ranges for stable libraries. Only flag pins that are genuinely
risky given the specific library's track record.

## Output format

Write the report to `/tmp/audit-{{suite}}.md`:

```markdown
<!-- agentic-ci-daily-{{suite}} -->
## Dependency Audit - {{date}}

### Transitive dependency gaps

| Package | Import | Declared in | Should be declared in |
|---------|--------|-------------|----------------------|
| engine | numpy | config only | engine (direct import in ...) |

### Cross-package inconsistencies

| Dependency | config pin | engine pin | interface pin | Issue |
|-----------|-----------|-----------|--------------|-------|
| ... | >=2.0,<3 | >=1.5 | (not declared) | Conflicting ranges |

### Unused dependencies

| Package | Dependency | Evidence |
|---------|-----------|----------|
| ... | ... | No imports found in src/ |

### Version pinning concerns

| Package | Dependency | Current pin | Concern |
|---------|-----------|-------------|---------|
| ... | litellm | >=1.0 | History of breaking changes; add upper bound |

### Summary

- N transitive gaps (M new since last run)
- N cross-package inconsistencies
- N unused dependencies (M new)
- N pinning concerns
```

If no findings in any category, write `NO_FINDINGS` on the first line instead.

## Constraints

- Do not modify any files. This is a read-only audit.
- Do not install packages or run `pip install`. Only inspect `pyproject.toml`
  and source files.
- Do not run `pip audit` (may not be available on the runner). Focus on
  structural dependency analysis, not CVE scanning (Dependabot handles that).
- Do not recommend changes to dependencies you haven't verified are actually
  problematic. False positives erode trust in the audit.
