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
find packages/*/tests/ -name 'test_*.py' 2>/dev/null | sort
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
grep -rn "assert .* is not None$" packages/*/tests/ --include='*.py'

# Test functions with no assert statements at all
grep -l "def test_" packages/*/tests/ --include='*.py' -r | while read f; do
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

The repo has a 3-second import budget tested by
`packages/data-designer/tests/test_import_perf.py`. Check the current state:

```bash
# Verify the test exists and read its thresholds
cat packages/data-designer/tests/test_import_perf.py 2>/dev/null || echo "not found"
```

Also check for heavy imports that bypass the lazy loading system:
```bash
# Direct imports of known heavy libraries at module level
grep -rn "^import pandas\|^from pandas\|^import numpy\|^from numpy\|^import duckdb\|^from duckdb\|^import faker\|^from faker" \
  packages/*/src/ --include='*.py'
```

These should use `data_designer.lazy_heavy_imports`. Report the count of
violations found, then refer to Wednesday's structure audit for the full
breakdown. Do not duplicate the detailed analysis here.

### 4. Executable smoke checks

Run lightweight checks that exercise real code paths. These catch silent
data corruption, column registration gaps, and config wiring issues that
static analysis misses. None of these require an LLM provider.

**Important**: the workflow puts `.venv/bin` on PATH via `make install-dev`,
so `python` resolves to the project venv with all packages installed.

There are two kinds of smoke checks: **fixed canaries** that must run
identically every time (deterministic regressions), and **creative checks**
where you should vary the inputs each run to maximize coverage over time.

#### Fixed canaries (run these exactly as written)

**4a. Package import verification**

```bash
python -c "
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.interface.data_designer import DataDesigner
print('OK: all packages import')
"
```

If any import fails, this is a critical finding - it means the package
layering is broken.

**4b. Import performance timing**

```bash
python -c "
import time
start = time.monotonic()
from data_designer.interface.data_designer import DataDesigner
elapsed = time.monotonic() - start
budget = 3.0
status = 'OK' if elapsed < budget else 'FAIL'
print(f'{status}: import took {elapsed:.2f}s (budget: {budget:.0f}s)')
"
```

**4c. Column type registry completeness**

```bash
python -c "
from data_designer.config.column_types import (
    DataDesignerColumnType,
    get_column_config_cls_from_type,
)

missing = []
for ct in DataDesignerColumnType:
    try:
        cls = get_column_config_cls_from_type(ct)
        if cls is None:
            missing.append(ct.value)
    except Exception as e:
        missing.append(f'{ct.value} ({e})')

if missing:
    for m in missing:
        print(f'FAIL: {m}')
else:
    print(f'OK: all {len(list(DataDesignerColumnType))} column types resolve to config classes')
" 2>&1 || echo "WARN: registry check could not run"
```

#### Creative checks (vary these each run)

For each run, **design your own** config build and validation checks. The
goal is to exercise different code paths over time rather than testing the
same config every day.

**What to vary:**
- **Sampler types**: pick a different mix each run. Available sampler types:
  `uuid`, `category`, `subcategory`, `uniform`, `gaussian`, `bernoulli`,
  `bernoulli_mixture`, `binomial`, `poisson`, `scipy`, `person_from_faker`,
  `datetime`, `timedelta`. Try 2-5 columns per config.
- **Column count**: sometimes build a single-column config, sometimes 8+
- **Edge cases**: empty params where defaults should apply, extreme param
  values (e.g., `gaussian` with `std=0`), columns with constraints
- **Recently changed code**: check `git log --oneline -20 -- packages/` for
  recently modified column types or sampler params, and prioritize testing
  those

**What to always check:**
1. Config build round-trip: column count and names survive `.build()`
2. Rejection: invalid inputs raise, not silently produce bad configs

**Limitation**: `DataDesigner(model_providers=[])` raises
`NoModelProvidersError`, so you cannot call `DataDesigner.validate()`
without at least one provider configured. Stick to config-layer checks
(`DataDesignerConfigBuilder.build()`, column type resolution) which do
not require providers.

**API reference** for writing checks:

```python
from data_designer.config.config_builder import DataDesignerConfigBuilder

# Build a config - use keyword args: name, column_type, sampler_type, params
builder = (
    DataDesignerConfigBuilder()
    .add_column(name='id', column_type='sampler', sampler_type='uuid')
    .add_column(name='cat', column_type='sampler', sampler_type='category',
                params={'values': ['A', 'B', 'C']})
)
config = builder.build()

# Verify columns survived the build
assert len(config.columns) >= 2
names = {c.name for c in config.columns}
assert 'id' in names and 'cat' in names
```

Run at least 2 creative checks per audit. Document what you chose and why
in the report (e.g., "tested poisson+datetime combo because poisson params
were modified in commit abc1234").

**Report smoke check results in a separate table.** If any check fails,
that is a higher-priority finding than static analysis results.

### 5. Test isolation verification

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

### Executable smoke checks

**Fixed canaries:**

| Check | Status | Detail |
|-------|--------|--------|
| Package imports | OK/FAIL | All three packages import cleanly |
| Import timing | OK/FAIL | X.XXs (budget: 3s) |
| Registry completeness | OK/WARN | Column types resolve to config classes |

**Creative checks** (describe what you tested and why):

| Check | What was tested | Status | Detail |
|-------|-----------------|--------|--------|
| Config build #1 | e.g. uuid+poisson+datetime | OK/FAIL | ... |
| Rejection #1 | e.g. gaussian with negative std | OK/FAIL | ... |
| ... | ... | ... | ... |

### Test isolation

| Test file | Violation |
|-----------|-----------|
| ... | Config test imports from engine |

### Summary

- N source files without tests (M new since last run)
- N hollow tests detected (high confidence only)
- Import perf: N heavy top-level imports
- Smoke checks: N passed, M failed (list any FAILs - these are critical)
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
