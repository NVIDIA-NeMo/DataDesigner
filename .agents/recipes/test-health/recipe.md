---
name: test-health
description: Audit test suite health - coverage gaps, hollow tests, import perf, test-to-source mapping
trigger: schedule
tool: claude-code
timeout_minutes: 20
max_turns: 30
permissions:
  contents: write
---

# Test Health Audit

Ensure the test suite stays meaningful, not just green. Write findings to
`/tmp/audit-{{suite}}.md`.

**What CI already enforces**: pytest with `--cov-fail-under=90` (aggregate),
multi-version matrix (3.10-3.13), multi-OS (ubuntu, macOS), per-package
isolation tests (config, engine, interface separately), e2e plugin tests.

**What CI does NOT catch**: per-file coverage regressions (aggregate can
stay above 90% while individual files lose coverage), hollow tests that
inflate coverage without testing behavior, import performance regressions
beyond the existing threshold, and new source files with no corresponding
tests.

## Runner memory

Read `{{memory_path}}/runner-state.json` for baselines from previous runs
(test-to-source mapping, import timing, known hollow tests). After the audit,
update `baselines` with current values and `known_issues` with new findings.

## Instructions

### 1. Test-to-source coverage mapping

Map source files to their corresponding test files:

```bash
# Source files (excluding __init__.py and test files)
find packages/*/src/ -name '*.py' -not -name '__init__.py' -not -path '*/test*' | sort

# Test files
find tests/ packages/*/tests/ -name 'test_*.py' 2>/dev/null | sort
```

For each source module, check if a corresponding test file exists. Flag:
- Source files with **no test file at all** (highest priority)
- New source files added since the last run that lack tests (compare against
  baseline in runner memory)

**Track the ratio**: N test files / M source files. Compare against baseline.

Focus on `packages/*/src/` only. Skip `scripts/`, `docs/`, and other
non-package code.

### 2. Hollow test detection

Scan test files for tests that assert nothing meaningful:

```bash
# Tests that only check "is not None" (often meaningless if the function
# can't return None)
grep -rn "assert .* is not None$" tests/ --include='*.py'

# Test functions with no assert statements at all
grep -l "def test_" tests/ --include='*.py' -r | while read f; do
  # Count test functions vs assert statements
  TESTS=$(grep -c "def test_" "$f")
  ASSERTS=$(grep -c "assert " "$f")
  if [ "$ASSERTS" -lt "$TESTS" ]; then
    echo "$f: $TESTS test functions, only $ASSERTS assertions"
  fi
done
```

**Be conservative**: only flag tests you've read and are confident add no
value. A test that looks simple may catch regressions that aren't obvious.
Read the test function body before flagging it.

Patterns that ARE hollow:
- `assert result is not None` where the return type is never Optional
- Test only verifies a mock was called, without checking the actual behavior
- Test calls a function and asserts nothing about the result

Patterns that are NOT hollow:
- `assert result is not None` as a guard before more specific assertions
- Tests that verify side effects (file creation, API calls, state changes)
- Tests that check exception behavior via `pytest.raises`

Skip `tests_e2e/` - e2e tests have different assertion patterns.

### 3. Import performance

The repo has a 3-second import budget tested by `tests/test_import_perf.py`.
Check the current state:

```bash
# Verify the test exists and read its thresholds
cat tests/test_import_perf.py 2>/dev/null || echo "not found"
```

Also check for heavy imports that bypass the lazy loading system:
```bash
# Direct imports of known heavy libraries at module level
grep -rn "^import pandas\|^from pandas\|^import numpy\|^from numpy\|^import duckdb\|^from duckdb\|^import faker\|^from faker" \
  packages/*/src/ --include='*.py'
```

These should use `data_designer.lazy_heavy_imports`. Cross-reference with
the structure recipe's findings if available in runner memory, but don't
skip this check - it directly affects user experience.

### 4. Test isolation verification

The CI runs three separate test jobs: config-only, engine+config, and
full stack. Check that test files respect these boundaries:

```bash
# Tests in packages/data-designer-config/tests/ should not import from engine
grep -rn "from data_designer\.engine\|import data_designer\.engine" \
  packages/data-designer-config/tests/ 2>/dev/null

# Tests in packages/data-designer-engine/tests/ should not import from interface
grep -rn "from data_designer\.interface\|import data_designer\.interface" \
  packages/data-designer-engine/tests/ 2>/dev/null
```

These would cause the isolated CI test jobs to fail, but catching them here
gives a clearer error message than a mysterious import failure.

## Output format

Write the report to `/tmp/audit-{{suite}}.md`:

```markdown
<!-- agentic-ci-daily-{{suite}} -->
## Test Health Audit - {{date}}

### Coverage gaps

| Source file | Test file | Status |
|------------|-----------|--------|
| engine/foo.py | (none) | No test file |

**Test-to-source ratio:** N test files / M source files (previous: X/Y)

### Hollow tests

| Test file | Test function | Issue | Confidence |
|-----------|--------------|-------|------------|
| ... | test_foo | Only asserts not None | high |

### Import performance

| Check | Status |
|-------|--------|
| test_import_perf.py exists | yes/no |
| Heavy top-level imports | N found |

### Test isolation

| Test file | Violation |
|-----------|-----------|
| ... | Config test imports from engine |

### Summary

- N source files without tests (M new since last run)
- N hollow tests detected (high confidence only)
- Import perf: N heavy top-level imports
- N test isolation violations
```

If no findings in any category, write `NO_FINDINGS` on the first line instead.

## Constraints

- Do not modify any test files. This is a read-only audit.
- Do not run the full test suite or coverage tool. Analysis is based on
  file structure and static inspection, not execution.
- Be conservative with hollow test detection. Only flag tests you've read
  and are confident add no value. Include confidence level in the report.
- Skip `tests_e2e/` from hollow test analysis.
- Do not duplicate the structure recipe's lazy import check in detail.
  Just flag the count and refer to Wednesday's structure audit for specifics.
