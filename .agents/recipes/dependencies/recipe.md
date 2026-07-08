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

This recipe also maintains `fix_backlog` and `attempted_fixes` per
`_fix-policy.md`. Update `fix_backlog` for every detected finding *before*
the `known_issues` filter applies.

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

### 2. Direct dependency declaration gaps

Read `/tmp/dependency-inventory.json`, generated deterministically before this
recipe by `scripts/audit_package_dependencies.py`. It uses Python ASTs and
installed distribution metadata to compare imports with static and dynamic
package declarations.

For every entry in `packages[].missing`:

- Re-read the listed source files to confirm the import is runtime code rather
  than a guarded optional import.
- Use `declared_by` to find a sibling package's existing version specifier.
- Treat `severity: low` entries as metadata hygiene: `guaranteed_by` names a
  mandatory workspace dependency that already installs the library. Do not
  claim these currently break standalone installation.
- Treat high severity as a candidate classification. Keep it high only after
  confirming the import is required runtime code; testing helpers and optional
  integrations stay report-only unless their packaging contract requires the
  dependency.

Review `unresolved_modules` manually. Report a gap only after mapping the
module to a distribution with repository evidence. Never guess a package name.

Only add a gap to `fix_backlog` when `declared_by` identifies a sibling
specifier that can be copied mechanically. Gaps without a sibling declaration
are report-only and must not consume the fix phase's top-five candidate limit.

Also check lazy imports in `data_designer/lazy_heavy_imports.py`; these are
intentionally deferred but still need to be declared as dependencies.

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

Use each inventory entry's `declared` and `imported` lists to seed unused
dependency candidates. The inventory is not proof of non-use: workspace
package dependencies, lazy imports, plugins, command entry points, and
runtime-only requirements may not appear as ordinary imports.

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

### Direct dependency declaration gaps

| Package | Import | Guaranteed by | Severity | Should be declared in |
|---------|--------|---------------|----------|-----------------------|
| engine | numpy | config | low | engine (direct import in ...) |

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

- N direct dependency declaration gaps (M new since last run)
- N cross-package inconsistencies
- N unused dependencies (M new)
- N pinning concerns
```

If no findings in any category, write `NO_FINDINGS` on the first line instead.

## Fix phase

Follow the standard fix procedure in `_fix-policy.md`. Suite-specific bits:

### Eligible categories

| Category | Branch type | test_required | Batchable | Eligibility note |
|----------|-------------|---------------|-----------|------------------|
| transitive-gap | `chore` | yes | yes, by target package (max 3) | Add the imported distribution to the dependency list of the package that imports it, copying the version specifier from a sibling package that already declares it. Insert in alphabetical order; match existing quote/specifier style. **Ineligible** when no sibling package declares the dep — choosing a specifier from scratch is interpretive, not mechanical. Those findings stay report-only and surface for maintainer judgement. |
| unused | `chore` | yes | no | Remove the declaration. Eligible only when checks across the package's `src/`, lazy-import system, plugin entry points, and tests turn up zero references. |

`fix_backlog.data` should record: for transitive-gap, the target package,
dependency name, importing source files, sibling package whose specifier was
copied, `guaranteed_by`, severity, and test target. The recipe must record this
during the audit; the fix phase rejects entries with no sibling source. For
unused, record which other packages also declare the dependency.

Batch transitive gaps only when they target the same package and test target.
Apply at most three dependency declarations in one PR. Include one hidden
finding marker and one `attempted_fixes` entry per dependency.

After editing the package manifest, run `make install-dev`, commit the
regenerated `uv.lock`, and run `uv lock --check`. The PR must include
`uv.lock`; a manifest-only dependency PR is incomplete. Then run the
per-package test target (see `_fix-policy.md`). `make install-dev` is the only
sanctioned install command (no direct `pip install` or `uv pip install`).

**Not eligible** — stays report-only:

- Cross-package version reconciliation, version pinning concerns
  (judgement-heavy).
- CVE response (Dependabot's job).

## Constraints

- Outside the fix phase, this recipe is read-only — do not modify files.
- Within the fix phase, only modify `packages/*/pyproject.toml` and `uv.lock`.
  The repo-root `pyproject.toml` is forbidden.
- `make install-dev` is the only sanctioned install command. Do not
  invoke `pip install` or `uv pip install` directly.
- Do not run `pip audit` (may not be available on the runner). Focus on
  structural dependency analysis, not CVE scanning (Dependabot handles that).
- Do not recommend changes to dependencies you haven't verified are actually
  problematic. False positives erode trust in the audit.
- Version pinning changes are explicitly out of scope for the fix phase.
