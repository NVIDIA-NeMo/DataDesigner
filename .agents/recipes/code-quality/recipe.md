---
name: code-quality
description: Audit code quality gaps not covered by ruff - complexity trends, exception hygiene, type coverage, TODO aging
trigger: schedule
tool: claude-code
timeout_minutes: 20
max_turns: 50
permissions:
  contents: write
---

# Code Quality Audit

Catch quality drift that CI doesn't cover. Write findings to
`/tmp/audit-{{suite}}.md`.

**What CI already enforces** (do NOT duplicate):
- Ruff rules: W, F, I, ICN, PIE, TID, UP006, UP007, UP045
- Ruff format with 120-char line length, double quotes
- Test coverage >= 90% aggregate

**What CI does NOT enforce** (this recipe's focus):
- C901 cyclomatic complexity (not in ruff select)
- ANN type annotation completeness (not in ruff select)
- BLE001 bare except handling (not in ruff select)
- Google-style docstring format (D* rules not enabled)
- Complexity growth trends over time
- TODO/FIXME aging

## Runner memory

Read `{{memory_path}}/runner-state.json` for baselines from previous runs
(complexity scores, type coverage, TODO inventory). After the audit, update
`baselines` with current values and `known_issues` with new findings. Skip
re-reporting known issues. Flag metrics that are trending in the wrong
direction compared to the previous baseline.

This recipe also maintains `fix_backlog` and `attempted_fixes` per
`_fix-policy.md`. Update `fix_backlog` for every detected bare-except
finding *before* the `known_issues` filter applies. (Other categories
remain report-only and do not enter `fix_backlog`.)

The `draft_until_proven` flag in runner-state controls whether this
suite's PRs are opened as draft. Default `true` until a maintainer flips
it to `false`.

## Instructions

### 1. Complexity hotspots

Try ruff C901 first (may not be in the config but can be invoked directly):
```bash
ruff check packages/*/src/ --select C901 --output-format json 2>/dev/null || true
```

If ruff is not available or C901 produces no output, manually inspect the
largest source files for functions with:
- Deep nesting (3+ levels of if/for/try)
- Many branches (>5 if/elif chains)
- Long method bodies (>60 lines)

**Track trends**: compare against the previous run's baseline in runner
memory. A function at complexity 12 that was 8 last week is more concerning
than one that has been at 15 for months. Report the delta.

Focus on `packages/data-designer-engine/src/` (core execution) and
`packages/data-designer/src/data_designer/interface/` (public API) where
complexity tends to accumulate.

### 2. Exception hygiene

Check for patterns that violate the project's "errors normalize at
boundaries" principle (AGENTS.md):

```bash
# Bare except clauses (should use specific exception types).
# Catches both `except:` and `except BaseException:` — both swallow
# everything including KeyboardInterrupt and SystemExit.
grep -rnE "except\s*:|except\s+BaseException" packages/*/src/ --include='*.py' | grep -v "# noqa"

# Swallowed exceptions (except + pass/continue with no logging)
grep -rn -A1 "except" packages/*/src/ --include='*.py' | grep -B1 "pass$\|continue$"
```

The key principle: internal code should NOT leak raw third-party exceptions.
Module boundary functions (public API, entry points) should wrap external
exceptions in `data_designer` error types. Check:
- Functions in `packages/data-designer/src/` that catch third-party exceptions
  (httpx, pydantic, etc.) - are they re-raised as `data_designer` errors?
- Plugin loading code (`data_designer/plugins/`) - bare `except:` has been
  found here before

### 3. Type annotation coverage

The repo requires typed code (AGENTS.md: "all functions, methods, and class
attributes require type annotations") but has no ANN ruff rules enforcing
this. Check for gaps:

```bash
# Public functions missing return type annotations
grep -rn "def " packages/*/src/ --include='*.py' \
  | grep -v "-> " \
  | grep -v "def _" \
  | grep -v "__init__\|__repr__\|__str__\|__eq__\|__hash__" \
  | grep -v "test_"
```

Also check for `Any` usage that could be more specific:
```bash
grep -rn ": Any\| -> Any" packages/*/src/ --include='*.py'
```

**Track coverage percentage**: count public functions with full annotations
vs total public functions. Compare against previous baseline.

Known gap: `packages/data-designer-config/src/data_designer/custom_column.py`
and `packages/data-designer-config/src/data_designer/analysis/` have been
flagged before.

### 4. Executable quality checks

Run a few checks that exercise real code paths to catch regressions that
static analysis misses. The workflow puts `.venv/bin` on PATH via
`make install-dev`, so `python` resolves to the project venv.

#### 4a. Error type hierarchy (fixed - run as written)

Verify that the project's error types are importable and properly
structured. Silent breakage here means third-party exceptions leak to users:

```bash
python -c "
from data_designer.errors import DataDesignerError
assert issubclass(DataDesignerError, Exception), 'DataDesignerError must be an Exception'
print('OK: error hierarchy intact')
" 2>&1 || echo "WARN: error hierarchy check failed"
```

#### 4b. Input validation checks (creative - vary each run)

Verify the config builder rejects bad inputs rather than silently
producing corrupt configs. **Design your own invalid inputs each run**
to maximize coverage over time.

Examples of things to test (pick 2-3 per run, and invent new ones):
- Invalid `column_type` string (should raise)
- `column_type='sampler'` without `sampler_type` (should raise)
- Empty builder `.build()` (should handle gracefully)
- Duplicate column names (should raise or deduplicate clearly)
- Invalid sampler params (e.g., `gaussian` with negative `std`, `category`
  with empty `values` list)
- Column names with special characters or very long strings
- Recently changed validators (check `git log --oneline -10 -- packages/*/src/data_designer/config/`)

**API reference:**

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder

# Test that invalid input is rejected (not silently accepted)
try:
    DataDesignerConfigBuilder().add_column(
        name='x', column_type='nonexistent_type'
    ).build()
    print('FAIL: invalid column type was silently accepted')
except Exception as e:
    print(f'OK: invalid column type rejected ({type(e).__name__})')
```

The pattern: try something that should fail, print FAIL if it succeeds
silently, print OK if it raises. A FAIL means a validation regression
that could lead to silent data corruption.

Report what you tested and why. Any FAIL is a critical finding.

### 5. TODO/FIXME/HACK aging

Inventory markers with their git blame age:

```bash
grep -rn "TODO\|FIXME\|HACK" packages/*/src/ --include='*.py'
```

For each marker, get the commit date:
```bash
# Example: get blame date for a specific line
git blame -L 42,42 --date=short path/to/file.py
```

**Only flag items older than 30 days.** Recent TODOs are part of normal
development flow. For old items, include:
- File and line number
- The marker text
- Age in days
- The commit that introduced it (short SHA)

## Output format

Write the report to `/tmp/audit-{{suite}}.md`:

```markdown
<!-- agentic-ci-daily-{{suite}} -->
## Code Quality Audit - {{date}}

### Complexity hotspots

| File | Function | Complexity | Trend |
|------|----------|-----------|-------|
| ... | ... | C901: 18 | +3 since last run |

### Exception hygiene

| File | Line | Pattern | Recommendation |
|------|------|---------|----------------|
| plugins/plugin.py | 99 | bare except | Catch ImportError/ModuleNotFoundError |

### Type annotation coverage

| File | Function | Issue |
|------|----------|-------|
| custom_column.py | generate | Missing return type |

**Coverage:** ~X% of public functions fully annotated (previous: Y%)

### Executable quality checks

| Check | Type | Status | Detail |
|-------|------|--------|--------|
| Error hierarchy | fixed | OK/FAIL | DataDesignerError is properly structured |
| (describe input tested) | creative | OK/FAIL | (what was tested and why) |
| ... | creative | ... | ... |

### TODO/FIXME/HACK inventory

| File | Line | Marker | Age (days) | Commit |
|------|------|--------|-----------|--------|
| ... | ... | TODO: fix this | 45 | abc1234 |

**Aging items:** N markers older than 30 days (M new since last run)

### Summary

- N complexity hotspots (M trending up)
- N exception hygiene issues (M new)
- Type coverage: X% (delta: +/-N% from last run)
- Executable checks: N/2 passed (any FAIL is critical)
- N aging TODO/FIXME markers (M new)
```

If no findings in any category, write `NO_FINDINGS` on the first line instead.

## Fix phase

Follow the standard fix procedure in `_fix-policy.md`. Suite-specific bits:

### Eligible categories

| Category | Branch type | test_required | Eligibility note |
|----------|-------------|---------------|------------------|
| bare-except | `refactor` | yes | Replace `except:` / `except BaseException:` with the specific exception type. Eligible only when grep across the try-block confirms **exactly one** exception type is plausibly raised, verified by inspecting the called functions or imported library docs. Multiple plausible types → ineligible. Test files are excluded (different exception-handling standards). |

`fix_backlog.data` should record the proposed replacement exception type
and the grep evidence used to determine it. Within bare-except findings,
prefer ones in user-facing modules (`packages/data-designer/src/`) over
internal helpers (the ranking impact criterion handles this once
`data.user_facing` is set).

The PR body should include the before/after of the try-block plus the
grep evidence that justified the chosen exception type, and a note that
the PR is draft until landing rate is proven (ask reviewers to mark
ready-for-review if the change is correct).

**Draft mode**: this suite opens PRs as draft until a maintainer flips
`draft_until_proven` to `false` in runner-state, after at least two
non-draft PRs have landed clean. Bare-except narrowing is the most
inference-heavy fix in any suite (confidence 0.6); recipe judgement has
to be earned before promotion. Two-strike findings here are an
especially important signal — they suggest the detector is producing
false positives in an already-cautious category.

**Not eligible** — stays report-only:

- Complexity refactors, type annotation additions, exception hierarchy
  normalization (judgement-heavy).
- **TODO line deletion** — the audit's "looks done" judgement is not
  mechanical enough to delete code on. Deletion is forbidden.

## Constraints

- Outside the fix phase, this recipe is read-only — do not modify files.
- Within the fix phase, only modify paths in the suite's path allowlist
  (`packages/*/src/**/*.py`). Test files are excluded.
- **TODO line deletion is forbidden.** The audit phase still inventories
  TODOs, but the fix phase does not act on them.
- Bare-except narrowing is only eligible when the exception type is
  unambiguous. When in doubt, skip.
- Do not flag test files for type coverage or exception hygiene. Tests have
  different standards.
- Do not duplicate ruff checks (W, F, I, ICN, PIE, TID, UP*). Those are
  already enforced in CI.
- For complexity, focus on growth trends rather than absolute values.
- For TODOs, only flag items older than 30 days.
- For type annotations, focus on public API surface. Internal helpers with
  obvious types from context are lower priority.
