---
name: datadesigner-docs
description: >
  Maintain the NeMo Data Designer Fern docs site under fern/. Use for any
  documentation change. Triggered by: "edit docs", "add doc page", "update
  docs", "rename page", "fix broken link", "add redirect", "preview docs",
  "publish docs", "regenerate notebooks", "update dev note", any request
  that touches `fern/`.
---

# Data Designer Docs Maintenance

Unified skill for adding, updating, moving, and removing pages on the NeMo Data Designer Fern docs site.

Current URL: **`datadesigner.docs.buildwithfern.com/nemo/datadesigner`** (see `instances` in [`fern/docs.yml`](../../../fern/docs.yml)). Source of truth for everything user-facing is `fern/`.

## Scope Rule

**ALL doc edits happen under `fern/`.** The legacy `docs/` directory is the original MkDocs source. `docs/notebook_source/*.py` remains canonical for notebook code, but **do not add new top-level prose pages under `docs/`**. Concept pages, recipes, plugins, and Dev Notes prose live under `fern/versions/latest/pages/`.

## Versioning Model

`main` contains only the latest Fern authoring docs. Historical release snapshots live on the CI-managed `docs-website` branch and must not be edited on `main`.

```
fern/versions/
├── latest.yml          ← authoring nav file
└── latest/pages/       ← authoring MDX tree
```

`docs.yml` on `main` registers only `slug: latest`. When you edit docs, edit `latest/pages/` and `latest.yml`.

Dev Notes are patched into the published latest docs by CI. Add new posts to `latest/pages/devnotes/posts/` and `latest.yml` on `main`.

For future Fern-native releases, do not copy page trees by hand on `main`. The release workflow snapshots `latest` into a frozen version on `docs-website`.

## Layout at a Glance

```
fern/
├── README.md                  ← maintainer cheat sheet
├── docs.yml                   ← title, theme, versions:, redirects, custom-domain
├── fern.config.json           ← organization + fern-api version pin
├── main.css                   ← bundled NVIDIA theme CSS
├── assets/                    ← logos, favicon, recipe assets, devnote post images (shared)
├── images/                    ← /images/* references from MDX (mirrors docs/images/)
├── styles/                    ← per-component CSS (notebook-viewer, authors, metrics-table, …)
├── components/                ← React/JSX MDX components
│   ├── NotebookViewer.tsx     ← renders converted .ipynb cells with outputs
│   ├── Authors.tsx            ← devnote bylines (uses devnotes/authors-data.ts)
│   ├── MetricsTable.tsx       ← benchmark tables w/ best-value highlight
│   ├── TrajectoryViewer.tsx   ← multi-turn tool-call traces (research dev notes)
│   ├── BadgeLinks.tsx         ← header shields (license, github, etc.)
│   ├── Tag.tsx, CustomCard.tsx, CustomFooter.tsx
│   ├── notebooks/             ← gitignored per-tutorial *.json + *.ts output
│   └── devnotes/              ← .authors.yml, authors-data.ts, per-post trajectory data
├── scripts/
│   └── ipynb-to-fern-json.py  ← .ipynb → fern/components/notebooks/*.{json,ts}
└── versions/
    ├── latest.yml             ← authoring navigation tree
    └── latest/pages/          ← authoring MDX content
```

## URL Routing Rules

Fern's URL is computed from the **section/page titles in the active version YAML**, not the file path:

```
File system                                           Published URL
────────────────────────────────────────────────────  ────────────────────────────────────────
latest/pages/concepts/columns.mdx                     /concepts/columns
latest/pages/concepts/tool_use_and_mcp.mdx            /concepts/tool-use-and-mcp/overview
latest/pages/recipes/code_generation/text_to_sql.mdx  /recipes/code-generation/text-to-sql
latest/pages/devnotes/posts/text-to-sql.mdx           /dev-notes/text-to-sql-for-nemotron-super
```

Rules:
- **Section title → kebab-case slug**: `"Dev Notes"` → `dev-notes`, `"Code Generation"` → `code-generation`, `"Tool Use & MCP"` → `tool-use-and-mcp`.
- **Page title → kebab-case slug**: `page: Text-to-SQL for Nemotron Super` → `text-to-sql-for-nemotron-super` (the filename `text-to-sql.mdx` is irrelevant for routing).
- **Subdirectories in the file path are dropped** — `devnotes/posts/foo.mdx` becomes `/dev-notes/<page-title>` (no `/posts/`).

When in doubt, recompute by walking the page's position in the active version YAML and slugifying each title.

## Operations

### Add a Page

1. Pick the file location: `fern/versions/latest/pages/<subdir>/<filename>.mdx`. Underscores in filenames are fine — they don't affect the URL.
2. Write minimal frontmatter:

   ```mdx
   ---
   title: "<Page Title>"
   description: "One-line SEO description (or empty string)"
   ---

   # <Page Title>

   <content>
   ```

3. Add an entry under the appropriate `section:` in `fern/versions/latest.yml`:

   ```yaml
   - page: <Page Title>
     path: ./latest/pages/<subdir>/<filename>.mdx
   ```

4. The URL becomes `/<section-slug>/<page-title-slug>`. Update any cross-references in other MDX accordingly.
5. If the page is a Dev Note, add it to the `Dev Notes` section in `latest.yml`.

### Update a Page

Editing prose is straightforward — change the MDX, save. No mirror step (one canonical tree).

For title changes, also update the `page:` value in `latest.yml` *and* fix any incoming links pointing at the old slugified title.

### Move / Rename a Page (with Redirect)

1. `git mv` the file.
2. Update `path:` in `latest.yml`.
3. Update incoming links: `grep -rn "<old-slug>" fern/versions/latest/pages/`.
4. Add a redirect to `fern/docs.yml`:

   ```yaml
   redirects:
     - source: "/nemo/datadesigner/<old-section>/<old-slug>"
       destination: "/nemo/datadesigner/<new-section>/<new-slug>"
   ```

   Note redirects use the path-prefixed form because `instances[0].url` includes `/nemo/datadesigner`.

### Remove a Page

1. `grep -rn "<filename-or-slug>"` to find inbound links.
2. `git rm` the MDX file.
3. Remove the `- page:` entry from `latest.yml`.
4. Fix or remove all inbound links.
5. Add a redirect in `docs.yml` if the URL was public.

### Worked Example: Add a How-To Under Concepts

Request: *"Add a how-to about deduplicating generated rows."*

1. Create `fern/versions/latest/pages/concepts/deduplication.mdx`:

   ```mdx
   ---
   title: "Deduplication"
   description: "Strategies for removing duplicates from generated datasets"
   ---

   # Deduplication

   <content>
   ```

2. Add to `fern/versions/latest.yml` under the `Concepts` section:

   ```yaml
   - page: Deduplication
     path: ./latest/pages/concepts/deduplication.mdx
   ```

3. Published URL: `/concepts/deduplication` on latest.
4. `cd fern && fern check && fern docs dev` to verify.

### Worked Example: Rename with Redirect

Request: *"Rename `/concepts/seed-datasets` to `/concepts/input-datasets`."*

1. Update `page: Seed Datasets` → `page: Input Datasets` in `latest.yml`.
2. `git mv fern/versions/latest/pages/concepts/seed-datasets.mdx fern/versions/latest/pages/concepts/input-datasets.mdx` and update `path:`.
3. Add to `docs.yml`:

   ```yaml
   redirects:
     - source: "/nemo/datadesigner/concepts/seed-datasets"
       destination: "/nemo/datadesigner/concepts/input-datasets"
     - source: "/nemo/datadesigner/latest/concepts/seed-datasets"
       destination: "/nemo/datadesigner/latest/concepts/input-datasets"
   ```

4. `grep -rn "/concepts/seed-datasets" fern/versions/latest/pages/` and rewrite hits.

---

## Content Guidelines

DataDesigner uses **Fern-native MDX components directly**. Do not use GitHub `> [!NOTE]` or MkDocs `!!! note` syntax — neither renders.

### Callouts

| Purpose | Component |
|---|---|
| Neutral aside | `<Note>...</Note>` |
| Helpful tip | `<Tip>...</Tip>` |
| Informational callout | `<Info>...</Info>` |
| Warning | `<Warning>...</Warning>` |
| Error / danger | `<Error>...</Error>` |

```mdx
<Tip>
Recipes assume an OpenAI provider. Configure with `data-designer config list`.
</Tip>
```

### Cards

Landing/index pages use `<CardGroup cols={N}>` + `<Card title="..." icon="..." href="...">`. Icons are Font Awesome names (no `:material-...:` shortcodes — those are MkDocs and render as text).

```mdx
<CardGroup cols={3}>
  <Card title="Tutorials" icon="book-open" href="/tutorials/the-basics">
    Step-by-step notebooks
  </Card>
  <Card title="Recipes" icon="hat-chef" href="/recipes/recipe-cards">
    Ready-to-use examples
  </Card>
</CardGroup>
```

### Header Badges (Landing Pages Only)

For shields-style badges (license, GitHub, model providers) at the top of an index page, use `<BadgeLinks>` from `fern/components/BadgeLinks.tsx`:

```mdx
import { BadgeLinks } from "@/components/BadgeLinks";

<BadgeLinks
  badges={[
    { href: "https://github.com/...", src: "https://img.shields.io/badge/...", alt: "GitHub" },
  ]}
/>
```

This wraps a row of badge images in a flex container so they sit side-by-side rather than stacking.

### Page Layout

The landing pages use frontmatter `layout: overview` for a wider design with TOC + sidebar. Other valid values: `guide` (default), `reference` (full-width, no TOC), `page` (no TOC, no nav), `custom` (blank canvas). See [Fern docs](https://buildwithfern.com/learn/docs/configuration/page-level-settings#layout).

## Frontmatter

Match the PR #581 style — keep frontmatter minimal:

```yaml
---
title: "<Page Title>"        # required
description: ""              # required (may be empty string)
layout: overview             # optional — only on landing pages
---
```

Do **not** add `position:` (we use explicit nav order in the version YAML), `date:`, or `authors:` to frontmatter — Fern's runtime treats `authors:` as a JSX scope variable and explodes when a component tries to reference it. For dev notes, see "Dev Notes" below.

## Dev Notes (Blog Posts)

Dev notes live under `fern/versions/latest/pages/devnotes/posts/`. They use the dev-notes kit components: **`<Authors>`, `<MetricsTable>`, `<TrajectoryViewer>`, `<ExpandableCode>`, `<CustomCard>`** (sources in `fern/components/`, CSS in `fern/styles/`).

Add the page and card to `latest.yml`. Release snapshots are handled by CI on `docs-website`.

### Authors Byline

Author registry: `fern/components/devnotes/.authors.yml` (source of truth) + `fern/components/devnotes/authors-data.ts` (typed copy that `Authors.tsx` imports). Edit both together.

```mdx
---
title: "<Post Title>"
description: "<one-line summary>"
---

import { Authors } from "@/components/Authors";

# <Post Title>

<Authors ids={["dnathawani", "ymeyer"]} />

<post body>
```

**Pass IDs as an explicit array literal**, not `<Authors ids={authors} />` — the bare `authors` identifier doesn't exist in Fern's MDX render scope.

To add a new author:

1. Append to `fern/components/devnotes/.authors.yml`:
   ```yaml
   newauthor:
     name: New Author
     description: Researcher at NVIDIA
     avatar: https://avatars.githubusercontent.com/u/<id>?v=4
   ```
2. Add the same entry to `fern/components/devnotes/authors-data.ts` (no auto-sync).

### Benchmark Tables

Use `<MetricsTable>` instead of plain markdown tables when you have numeric model comparisons. Auto-highlights the best value per column.

```mdx
import { MetricsTable } from "@/components/MetricsTable";

<MetricsTable
  headers={["Model", "BIRD EX (%)", "Val Loss (↓)"]}
  rows={[
    ["Baseline", 26.77, 1.309],
    ["Ours", 41.80, 1.256],
  ]}
  lowerIsBetter={[2]}
/>
```

`lowerIsBetter` takes column indices; everything else is "higher is better".

### Trajectory Viewer

For multi-turn agent traces (search/open/find/answer), put the trajectory data in a typed module under `fern/components/devnotes/<post-slug>/<example>.ts`:

```ts
import type { TrajectoryViewerProps } from "@/components/TrajectoryViewer";

const trajectory: Omit<TrajectoryViewerProps, "defaultOpen"> = {
  question: "...",
  referenceAnswer: "...",
  goldenPassageHint: "⭐ = golden passage",
  summary: "Example: 31 turns, 49 tool calls",
  turns: [
    { turnIndex: 1, calls: [{ fn: "search", arg: "..." }] },
    // ...
    { turnIndex: 31, calls: [{ fn: "answer", body: "..." }] },
  ],
};

export default trajectory;
```

Then in the post:

```mdx
import { TrajectoryViewer } from "@/components/TrajectoryViewer";
import trajectory from "@/components/devnotes/<post-slug>/<example>";

<TrajectoryViewer {...trajectory} defaultOpen />
```

The component is collapsible when `summary` is set; uncollapsed when `defaultOpen` is true. **Don't** hand-code trajectory HTML/CSS in MDX — the original `<div class="trajectory-viz">` + `<style>` approach was migrated away because MDX runs the CSS body through acorn and explodes on `{ color: ... }`.

### `<ExpandableCode>` — caveat

The kit ships `ExpandableCode` for collapsible code dumps, **but it currently breaks Fern's SSR runtime** (the `onClick` + `setTimeout` in the copy button trips the React error boundary, page renders "Something went wrong!"). Use plain ` ```python ` fences for now. Component file is kept in `fern/components/` for future use if Fern's runtime support changes.

## Notebook Tutorials

Notebook tutorials are JS-driven via `<NotebookViewer>` rendering converted `.ipynb` data. The whole pipeline:

```
docs/notebook_source/*.py            (jupytext format — canonical source, edit here)
        │ make convert-execute-notebooks      # jupytext --execute (needs NVIDIA_API_KEY)
        ▼
docs/notebooks/*.ipynb               (executed; outputs captured)
        │
        │ make generate-fern-notebooks         # per-file prefers executed docs/notebooks/
        │                                      # otherwise converts notebook_source/*.py directly
        ▼
fern/components/notebooks/*.{json,ts} (gitignored; generated before preview/publish)
```

`docs/colab_notebooks/*.ipynb` is a separate committed output for "Open in Colab" links. It is generated by `make generate-colab-notebooks`, but it is not a Fern docs build input.

The `.ts` is what the wrapper MDX imports. Fern's bundler doesn't follow `.json` imports cleanly.

### Make Targets

| Command | When |
|---------|------|
| `make generate-fern-notebooks` | Notebook prose changed, no need to re-execute. Per file, prefers `docs/notebooks/` (executed) and falls back to converting `docs/notebook_source/*.py` directly. |
| `make generate-fern-notebooks-with-outputs` | Notebook code changed, want fresh outputs. Needs `NVIDIA_API_KEY` (and `OPENROUTER_API_KEY` for image notebooks 5–6). |

Install notebook docs dependencies first with `make install-dev-notebooks`. Docs setup pins to `DOCS_PYTHON_VERSION ?= 3.13` because `pyarrow` lacks Python 3.14 wheels. Override via `DOCS_PYTHON_VERSION=3.12 make ...`.

The `convert-execute-notebooks` step loops per file so one notebook missing an API key does not prevent later notebooks from running. Any failure is reported after the loop and the make target exits non-zero.

### Wrapper Page

Each `notebooks/<name>.mdx` wrapper is tiny:

```mdx
---
title: "The Basics"
description: "Declare columns, generate your first dataset."
---

import { NotebookViewer } from "@/components/NotebookViewer";
import notebook from "@/components/notebooks/1-the-basics";

<NotebookViewer
  notebook={notebook}
  colabUrl="https://colab.research.google.com/github/NVIDIA-NeMo/DataDesigner/blob/main/docs/colab_notebooks/1-the-basics.ipynb"
/>
```

The converter (`fern/scripts/ipynb-to-fern-json.py`) **auto-strips the leading Colab badge cell** — `<NotebookViewer>` renders its own banner from the `colabUrl` prop. Don't manually re-add it.

## MDX Gotchas (the ones that bit during migration)

| Pattern | Problem | Fix |
|---------|---------|-----|
| `{ width=800 }`, `{ .md-button }`, `{ target="_blank" }` after a link/image | MkDocs Material `attr_list`; MDX parses `{...}` as JSX | Strip them — Fern doesn't support these annotations |
| `--8<-- "path/to/file.py"` inside a code fence | pymdown `snippets` syntax | Inline the file contents directly |
| `:material-icon-name:`, `:octicons-...:` | pymdown emoji shortcodes | Render as literal text in Fern; replace with Font Awesome `icon=` on `<Card>` or remove |
| Bare `{"key": "value"}` JSON in prose | MDX tries to parse as JSX expression | Wrap in inline backticks: `` `{"key": "value"}` `` |
| `<style>{ /* CSS */ }</style>` raw | MDX runs CSS body through acorn | Wrap in template literal: `<style>{`/* css */`}</style>` |
| `<br>`, `<img src="...">` (unclosed void elements) | MDX requires self-closing | Convert to `<br />`, `<img src="..." />` |
| `<4`, `<10MB` in prose | MDX reads as JSX tag | Escape: `&lt;4`, `&lt;10MB` |
| `<!-- HTML comment -->` | Not valid MDX | Use `{/* MDX comment */}` |
| `<Authors ids={authors} />` referencing frontmatter `authors:` | `authors` is not in JSX scope | Pass an explicit literal: `<Authors ids={["jdoe"]} />` |

## Validate

Run from `fern/`:

```bash
fern check          # YAML + frontmatter + MDX validation
fern docs dev       # localhost:3000 hot-reload preview
```

`fern check` must pass before commit. The local broken-link checker has known false positives — it computes URLs from file paths instead of from slugified nav titles, so cross-section absolute links sometimes flag incorrectly. Spot-check by clicking through the dev server.

## Commit & Preview

```bash
git add fern/
git commit -s -m "docs: <add|update|remove> <page-title>"
```

DCO sign-off (`-s`) is required by CONTRIBUTING. Use `docs:` prefix (matches recent commit history). Subject line ≤ 50 chars (hard limit 72).

When the team adds a Fern preview workflow (modeled after Gym's `fern-docs-preview-comment.yml`), PRs touching `fern/**` will get an automatic preview URL posted as a comment. Until that lands, share local dev-server screenshots in PR descriptions.

## Cutting a New Version Train

Do not copy page trees by hand on `main`. The release workflow copies `latest/pages/` to a frozen `vX.Y.Z/pages/` tree on `docs-website`, updates the published versions list, checks the generated Fern docs, and publishes from `docs-website`.

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML |
| Page 404 in preview | Section/page title mismatch — recompute slug from `latest.yml`, not from filename |
| `Could not parse expression with acorn` | Bare `{...}` JSON or pymdown attr_list; see MDX Gotchas |
| `Could not parse import/exports with acorn` | Missing blank line between top-level `import` statement and `# H1` body |
| `Unexpected character 'X' before name` | Likely `--8<--` snippet include or bare `<digit`; see MDX Gotchas |
| "Something went wrong!" runtime error | A custom component is throwing — check `<Authors ids={authors} />` (use literal array) or `<ExpandableCode>` (currently broken in SSR) |
| Notebook page renders raw `<a href=colab...>` HTML | `.ts` was generated before the colab-strip improvement; re-run `make generate-fern-notebooks` |
| Notebook page has no cell outputs | Ran without `NVIDIA_API_KEY` or `convert-execute-notebooks` failed; run `make generate-fern-notebooks-with-outputs` |
| `URLError: [SSL: CERTIFICATE_VERIFY_FAILED]` during notebook execution | `DOCS_CERTS` not propagated; ensure you're invoking via the make target, not raw Python |
| `Failed to build pyarrow==X` from source | `DOCS_PYTHON_VERSION` resolved to 3.14+; override with `DOCS_PYTHON_VERSION=3.13 make ...` (or just rely on the default) |
| Cards on landing all link to the same wrong URL | `href` not matching Fern's slugified-title rule — recompute as `/<section-slug>/<page-title-slug>` |
| Image broken in preview, file exists at `fern/assets/...` | Reference uses relative `../assets/...` — change to absolute `/assets/...` (relative paths break across version slugs) |

## When NOT to Use This Skill

- Editing Python source under `packages/` — that's a code change, not a docs change.
- Adding a notebook tutorial's *code*: edit `docs/notebook_source/<name>.py`, not the converted `.ipynb` or the wrapper MDX.
- Editing dev note *prose*: edit the migrated MDX under `fern/versions/latest/pages/devnotes/posts/<name>.mdx`. (The original `docs/devnotes/posts/<name>.md` is no longer the source of truth — Fern is.)
