# Dev Notes Custom Components Plan

Plan for Fern custom React components that replicate the experience of `docs/devnotes/` pages. Use this when Fern enables custom components (Pro/Enterprise).

## ✅ Implemented Components

| Component | File | Status |
|-----------|------|--------|
| TrajectoryViewer | `components/TrajectoryViewer.tsx` | Done |
| ExpandableCode | `components/ExpandableCode.tsx` | Done |
| PipelineDiagram | `components/PipelineDiagram.tsx` | Done |
| MetricsTable | `components/MetricsTable.tsx` | Done |
| 4hop trajectory data | `components/trajectories/4hop-example.ts` | Done |
| SDG pipeline diagram | `components/diagrams/sdg-pipeline.ts` | Done |

## Current Dev Notes Structure

| File | Content Type | Key Elements |
|------|--------------|--------------|
| `index.md` | Landing page | Intro, auto-listing of posts |
| `posts/rqa.md` | Blog post | Hero image, code blocks, comparison tables |
| `posts/design-principles.md` | Blog post | Hero image, ASCII pipeline diagram, code blocks |
| `posts/deep-research-trajectories.md` | Blog post | Hero image, **TrajectoryViewer** (custom HTML/CSS), code blocks, expandable source, tables |
| `.authors.yml` | Metadata | Author name, description, avatar |

---

## Component Inventory

### 1. **TrajectoryViewer** (High Priority)

**Purpose:** Renders multi-turn research trajectories with tool calls (search, open, find) in a structured, color-coded layout.

**Current implementation:** ~100 lines of inline HTML/CSS in `deep-research-trajectories.md`.

**Props:**
```ts
interface TrajectoryViewerProps {
  question: string;
  referenceAnswer?: string;
  goldenPassageHint?: string;  // e.g. "⭐ = golden passage"
  turns: TrajectoryTurn[];
  defaultOpen?: boolean;       // For collapsible wrapper
  summary?: string;            // e.g. "Example trajectory: 4-hop question, 31 turns"
}

interface TrajectoryTurn {
  turnIndex: number;
  calls: ToolCall[];
}

interface ToolCall {
  fn: "search" | "open" | "find" | "answer";
  arg: string;
  isGolden?: boolean;          // For open() calls with ⭐
  body?: string;               // For "answer" - full answer text (supports HTML)
}
```

**Visual design:**
- Question: blue-tinted background (`#42a5f5`)
- Reference: green-tinted, left border
- search: blue accent
- open: green accent
- find: orange accent
- answer: green, full-width block
- Turn labels: T1, T2, ... monospace, muted
- Parallel calls: grouped with vertical bar

**Data format:** Accept structured JSON or a simplified DSL. Could also accept pre-rendered HTML for migration.

---

### 2. **PipelineDiagram** (Medium Priority)

**Purpose:** Renders ASCII/text pipeline diagrams with consistent styling (e.g., design-principles SDG stages).

**Current implementation:** Raw ASCII in markdown:
```
      Seed Documents         Seed dataset column ingests documents
            │                 from local files or HuggingFace
            ▼
┌─────────────────────────┐
│  Artifact Extraction    │  LLM extracts key concepts...
```

**Props:**
```ts
interface PipelineDiagramProps {
  /** ASCII diagram string - preserves whitespace, monospace font */
  diagram: string;
  /** Optional title/caption */
  title?: string;
  /** Max width for horizontal scroll */
  maxWidth?: string;
}
```

**Implementation:** Monospace block, optional syntax highlighting for box-drawing chars, scroll on overflow.

---

### 3. **ExpandableCode** (Medium Priority)

**Purpose:** Collapsible code block with "Full source" summary. Used in deep-research-trajectories for `openresearcher_demo.py`, `prepare_corpus.py`, `retriever_mcp.py`.

**Props:**
```ts
interface ExpandableCodeProps {
  summary: string;            // e.g. "Full source: openresearcher_demo.py"
  code: string;
  language?: string;          // python, etc.
  defaultOpen?: boolean;
}
```

**Implementation:** Wraps Fern's built-in code block in a `<details>`-like collapsible. Uses Fern's Accordion if available, or custom collapse.

---

### 4. **MetricsTable** (Low–Medium Priority)

**Purpose:** Styled comparison tables for benchmark results (e.g., RQA validation loss, MMLU-Pro, GSM8K).

**Current implementation:** Standard markdown tables with `:----:` alignment.

**Props:**
```ts
interface MetricsTableProps {
  headers: string[];
  rows: (string | number)[][];
  /** Optional: highlight best values per column (e.g. bold) */
  highlightBest?: "min" | "max" | "none";
  /** Column indices where lower is better (e.g. validation loss) */
  lowerIsBetter?: number[];
}
```

**Enhancement:** Could auto-bold best values, add subtle zebra striping. Fern may already style tables well—verify first.

---

### 5. **DevNoteCard** (Low Priority)

**Purpose:** Article preview card for dev notes index—title, date, authors, excerpt (content before `<!-- more -->`).

**Props:**
```ts
interface DevNoteCardProps {
  title: string;
  slug: string;
  date: string;
  authors: string[];          // Keys from .authors.yml
  excerpt: string;
  image?: string;
}
```

**Dependency:** Requires authors data. Could be passed as prop or loaded from a generated JSON.

---

### 6. **AuthorCard** (Low Priority)

**Purpose:** Renders author info (avatar, name, description) from `.authors.yml`.

**Props:**
```ts
interface AuthorCardProps {
  authorId: string;           // dcorneil, etramel, kthadaka, nvidia
  name?: string;
  description?: string;
  avatar?: string;
}
```

**Note:** Fern may have built-in author/avatar support. Check before implementing.

---

### 7. **HeroImage** (Low Priority)

**Purpose:** Full-width or aligned hero image with optional caption. Used at top of each dev note.

**Current usage:** `![alt](url){ align=right width=500 }` (MkDocs Material syntax).

**Props:**
```ts
interface HeroImageProps {
  src: string;
  alt: string;
  align?: "left" | "center" | "right";
  width?: number | string;
  caption?: string;
}
```

**Note:** Fern's image handling may suffice. Verify if custom alignment/caption is needed.

---

### 8. **ResourceLinks** (Low Priority)

**Purpose:** Styled "Key Resources" section at end of articles (numbered list with links).

**Current implementation:** Plain markdown list.

**Props:**
```ts
interface ResourceLinksProps {
  items: { label: string; href: string }[];
  title?: string;             // Default: "Key Resources"
}
```

---

## Implementation Order

| Phase | Component | Effort | Impact |
|-------|-----------|--------|--------|
| 1 | **TrajectoryViewer** | High | Critical for deep-research-trajectories |
| 2 | **ExpandableCode** | Low | Used in 3 places in deep-research |
| 3 | **PipelineDiagram** | Low | design-principles only |
| 4 | **MetricsTable** | Low | Nice-to-have for RQA, design-principles |
| 5 | **DevNoteCard**, **AuthorCard**, **HeroImage**, **ResourceLinks** | Low | Index page, polish |

---

## Migration Strategy

1. **Create components** in `fern/components/` following NotebookViewer pattern.
2. **Add CSS** to `fern/styles/` (e.g. `trajectory-viewer.css`).
3. **Extract trajectory data** from deep-research-trajectories.md into a JSON/TS file (like notebooks).
4. **Convert MDX** – replace inline HTML with component usage:
   ```mdx
   import { TrajectoryViewer } from "@/components/TrajectoryViewer";
   import exampleTrajectory from "@/components/trajectories/4hop-example";

   <TrajectoryViewer
     question={exampleTrajectory.question}
     referenceAnswer={exampleTrajectory.referenceAnswer}
     turns={exampleTrajectory.turns}
     summary="Example trajectory: 4-hop question, 31 turns, 49 tool calls"
     defaultOpen
   />
   ```
5. **Register** in `docs.yml` under `experimental.mdx-components`.

---

## Data Extraction Tasks

| Source | Output | Format |
|--------|--------|--------|
| deep-research-trajectories.md (lines 141–174) | `trajectories/4hop-example.ts` | `{ question, referenceAnswer, turns }` |
| design-principles.md (pipeline ASCII) | Inline or `diagrams/sdg-pipeline.ts` | `{ diagram: string }` |

---

## Fern Compatibility Notes

- **No React import** – use automatic JSX runtime (see NotebookViewer).
- **No class components** – ErrorBoundary etc. not available.
- **Server-side rendered** – components must work without client-side JS.
- **CSS** – add to `docs.yml` `css:` array.
- **Pro/Enterprise** – custom components require Fern Pro or Enterprise plan.

---

## Open Questions

1. **Dev notes section** – Will dev notes live under a new Fern section (e.g. "Dev Notes" tab) or within existing structure?
2. **Authors** – How to integrate `.authors.yml`? Build-time script to generate author lookup? Or hardcode in MDX frontmatter?
3. **Trajectory data** – Manual extraction vs. script to parse HTML? Manual is fine for one example; script if we add more.
4. **Blog index** – Does Fern support blog-style listing (chronological posts)? Or do we use a static "Dev Notes" page with manual card grid?
