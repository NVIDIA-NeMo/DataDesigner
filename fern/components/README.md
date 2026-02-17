# Fern Components

Custom React components for NeMo Data Designer docs. Fern loads them via `mdx-components` in `docs.yml`; use `@/components/...` imports in MDX.

## Components

### Authors

Author byline with avatars for dev notes.

**When to use:** Top of dev note pages.

```mdx
import { Authors } from "@/components/Authors";

<Authors ids={["dcorneil", "etramel"]} />
```

**Data:** `devnotes/authors-data.ts` (synced with `devnotes/.authors.yml`). Add new authors there.

---

### MetricsTable

Styled comparison table with optional best-value highlighting.

**When to use:** Benchmark results, before/after comparisons.

```mdx
import { MetricsTable } from "@/components/MetricsTable";

<MetricsTable
  headers={["Data Blend", "Validation Loss", "MMLU-Pro"]}
  rows={[
    ["Baseline", "1.309", "36.99"],
    ["with RQA (4%)", "1.256", "44.31"],
  ]}
  lowerIsBetter={[1]}
/>
```

**Props:** `headers`, `rows`, `lowerIsBetter` (column indices), `higherIsBetter`.

---

### TrajectoryViewer

Multi-turn research trajectories with tool calls (search, open, find, answer).

**When to use:** Deep research / MCP tool-use dev notes.

```mdx
import { TrajectoryViewer } from "@/components/TrajectoryViewer";
import trajectory from "@/components/devnotes/deep-research-trajectories/4hop-example";

<TrajectoryViewer {...trajectory} defaultOpen />
```

**Data:** Define trajectory objects in `devnotes/<post-name>/` and import. See `4hop-example.ts` for shape.

---

### NotebookViewer

Renders Jupyter notebook cells (markdown + code) with optional Colab badge.

**When to use:** Tutorial pages. Data comes from `make generate-fern-notebooks` (see `fern/README.md`).

```mdx
import { NotebookViewer } from "@/components/NotebookViewer";
import notebook from "@/components/notebooks/1-the-basics";

<NotebookViewer
  notebook={notebook}
  colabUrl="https://colab.research.google.com/github/NVIDIA-NeMo/DataDesigner/blob/main/docs/colab_notebooks/1-the-basics.ipynb"
/>
```

**Props:** `notebook`, `colabUrl`, `showOutputs` (default true).

---

### ExpandableCode

Collapsible code block with summary and copy button.

**When to use:** Long code snippets you want collapsed by default (e.g. "Full source" in dev notes).

```mdx
import { ExpandableCode } from "@/components/ExpandableCode";

<ExpandableCode
  summary="Full source: openresearcher_demo.py"
  code={`...`}
  language="python"
  defaultOpen={false}
/>
```

---

### CustomCard

Simple card with title, text, and link. Alternative to Fern's built-in `Card` when you need custom styling.

**When to use:** Tutorial overview or landing pages when `Card` doesn't fit.

```mdx
import { CustomCard } from "@/components/CustomCard";

<CustomCard title="Title" text="Description" link="/path" sparkle />
```

---

## Subdirectories

| Path | Purpose |
|------|---------|
| `devnotes/` | Author data (`.authors.yml`, `authors-data.ts`), post-specific assets (e.g. `deep-research-trajectories/4hop-example.ts`) |
| `notebooks/` | Generated notebook data (`*.json`, `*.ts`). Created by `make generate-fern-notebooks`. Do not edit by hand. |

## Adding a New Component

1. Create `ComponentName.tsx` in `components/`.
2. Use Fern's automatic JSX runtime (no `import React`).
3. Add corresponding CSS in `fern/styles/` and register in `docs.yml` under `css:`.
4. Import in MDX: `import { ComponentName } from "@/components/ComponentName";`
