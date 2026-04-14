---
name: structure
description: Audit structural integrity - import boundaries, lazy import compliance, future annotations, dead exports
trigger: schedule
tool: claude-code
timeout_minutes: 20
max_turns: 30
permissions:
  contents: write
---

# Structural Integrity Audit

Verify the multi-package layering that DataDesigner depends on. Write findings
to `/tmp/audit-{{suite}}.md`.

## Background

Before starting, read the authoritative sources for the structural rules:

1. **`AGENTS.md`** (repo root) - "The Layering Is Structural" section defines
   the three packages, their ownership, and the dependency direction rule.
2. **`architecture/overview.md`** - system architecture diagram, package layout,
   and the "no reverse imports" rule.

The canonical rules:

```
data-designer (interface) -> data-designer-engine -> data-designer-config
```

- `packages/data-designer-config/` must NOT import from `data_designer.engine`,
  `data_designer.interface`, or `data_designer.cli`
- `packages/data-designer-engine/` must NOT import from
  `data_designer.interface` or `data_designer.cli`
- `packages/data-designer/` CAN import from both engine and config

**What CI already enforces**: ruff rule `TID` catches relative imports. The
CI test matrix runs config, engine, and interface tests in isolation (separate
jobs with only the package's own deps installed), which catches most boundary
violations at test time.

**What CI does NOT catch**: violations that only manifest with all packages
installed (e.g., an engine file importing from interface, which works when
interface is installed but fails in the isolated engine test only if that
code path is exercised). This recipe does static analysis to catch these
regardless of test coverage.

## Runner memory

Read `{{memory_path}}/runner-state.json` for known issues from previous runs.
Update after the audit. Skip re-reporting known issues.

## Instructions

### 1. Import boundary violations

Scan each package's source for imports that violate the dependency direction:

```bash
# Config must not import from engine or interface
grep -rn "from data_designer\.engine\|import data_designer\.engine\|from data_designer\.interface\|import data_designer\.interface\|from data_designer\.cli\|import data_designer\.cli" \
  packages/data-designer-config/src/

# Engine must not import from interface
grep -rn "from data_designer\.interface\|import data_designer\.interface\|from data_designer\.cli\|import data_designer\.cli" \
  packages/data-designer-engine/src/
```

**Important**: exclude `TYPE_CHECKING` blocks. Imports guarded by
`if TYPE_CHECKING:` are allowed since they don't execute at runtime. Read
the surrounding context of each match to check.

As of the last audit, import boundaries were clean. If this section has no
findings, that's expected - it's a guardrail, not a bug finder.

### 2. Lazy import compliance

The repo uses `data_designer.lazy_heavy_imports` for heavy third-party
libraries (AGENTS.md: "heavy third-party libraries are lazy-loaded").
Check for direct top-level imports that bypass the lazy system:

```bash
# Known heavy libraries that should use lazy imports
grep -rn "^import pandas\|^from pandas\|^import numpy\|^from numpy\|^import polars\|^from polars\|^import torch\|^from torch\|^import duckdb\|^from duckdb\|^import sqlfluff\|^from sqlfluff\|^import faker\|^from faker" \
  packages/*/src/ --include='*.py'
```

Exclude:
- The lazy import system itself (`lazy_heavy_imports.py`)
- `TYPE_CHECKING` blocks
- Test files
- Files that are themselves optional/heavy modules

This directly affects startup time, which has a 3-second budget tested by
`packages/data-designer/tests/test_import_perf.py`.

### 3. Future annotations compliance

AGENTS.md requires `from __future__ import annotations` in every Python
source file:

```bash
find packages/*/src/ -name '*.py' -not -name '__init__.py' | while read f; do
  if ! grep -q "from __future__ import annotations" "$f"; then
    echo "$f"
  fi
done
```

This enables modern type syntax (`list[str]` instead of `List[str]`,
`str | None` instead of `Optional[str]`) and defers annotation evaluation.

### 4. Graphify structural analysis

Run the graphify-based structural analysis tool. This builds a directed AST
graph of the entire codebase (~2s) and produces god node rankings,
cross-package edge counts, and import direction violations that grep-based
checks miss (e.g., inferred cross-file dependencies from class-level import
resolution).

```bash
PREV_GRAPH=""
if [ -f "graphify-out/graph.json" ]; then
  PREV_GRAPH="--previous-graph graphify-out/graph.json"
fi
python .agents/tools/structural_impact.py --full $PREV_GRAPH \
  --output /tmp/graphify-structure.md
```

Read `/tmp/graphify-structure.md` and incorporate relevant findings:

- **God nodes**: track the top 10 most-connected entities. Compare against
  the previous baseline in runner memory. A god node gaining 10+ connections
  since last week signals growing coupling.
- **Cross-package edge summary**: the table shows edge counts per direction.
  All edges should flow interface -> engine -> config. Any VIOLATION row is a
  finding for the import boundary section.
- **Changes since last run**: new/removed entities and relationships since
  the previous audit. Flag any unexpected removals of god nodes.

After the audit, update `baselines` in runner memory with the current god
node list and cross-package edge counts from `graphify-out/baselines.json`.

### 5. Dead exports

Find symbols in `__all__` that nothing outside their module references:

```bash
grep -rn "__all__" packages/*/src/ --include='*.py'
```

For each symbol in `__all__`:
- Search the codebase for imports of that symbol
- If only referenced within its own file, flag as potentially dead

**Be conservative**: symbols in `__all__` of top-level `__init__.py` files
are part of the public API and may be used by external consumers (plugins,
user code). Only flag symbols that are clearly internal and unused.

## Output format

Write the report to `/tmp/audit-{{suite}}.md`:

```markdown
<!-- agentic-ci-daily-{{suite}} -->
## Structural Integrity Audit - {{date}}

**Rules checked against:** AGENTS.md, architecture/overview.md

### Import boundary violations

| Package | File | Line | Import | Rule violated |
|---------|------|------|--------|---------------|
| ... | ... | ... | ... | config -> engine (forbidden per AGENTS.md) |

### Lazy import violations

| File | Line | Import | Should use |
|------|------|--------|-----------|
| ... | ... | import pandas | lazy_heavy_imports |

### Missing future annotations

| File |
|------|
| ... |

### Dead exports

| Package | Module | Symbol | Evidence |
|---------|--------|--------|----------|
| ... | ... | ... | No external imports found |

### Structural graph analysis (graphify)

(Paste the content of /tmp/graphify-structure.md here)

### Summary

- N import boundary violations (M new since last run)
- N lazy import violations (M new)
- N files missing future annotations (M new)
- N potentially dead exports (M new)
- God nodes: top entity has N connections (delta: +/-M from last run)
- Cross-package edges: N total (M violations)
```

If no findings in any category, write `NO_FINDINGS` on the first line instead.

## Constraints

- Do not modify any files. This is a read-only audit.
- Imports inside `if TYPE_CHECKING:` blocks are allowed and should not be
  flagged for any check.
- Lazy imports in `__init__.py` (via `__getattr__`) are deferred and should
  not be treated as violations.
- Dead export detection has false positives. Mark uncertain cases as
  "potentially dead" rather than definitively dead.
- Always cite which rule or doc is violated so maintainers can verify.
- Import boundaries are currently clean. No findings in that section is
  normal and expected.
