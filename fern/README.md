# Fern Docs

This folder is the Fern Docs build for NeMo Data Designer. The site is published at **`docs.nvidia.com/nemo/datadesigner`** (configured in [`docs.yml`](docs.yml)).

## Prerequisites

```bash
# Install Fern CLI globally
npm install -g fern-api
```

## First-time setup

Two pre-render steps are needed before the dev server has all content. Both produce files that are gitignored OR live alongside committed snapshots — running them is idempotent.

### 1. Python API reference (gitignored — must regenerate)

The `libraries:` block in [`docs.yml`](docs.yml) tells Fern to extract API docs from this repo's Python source (`packages/data-designer-config/src/data_designer/config`). The output lands in `fern/code-reference/data-designer/` (gitignored).

```bash
cd fern
fern docs md generate     # no FERN_TOKEN required; clones from GitHub + runs Pyright
```

Re-run when the upstream package source changes.

### 2. Notebook tutorials (committed snapshots — regenerate on edit)

Each tutorial `.ipynb` is converted to a JSON+TS pair in `fern/components/notebooks/`, then rendered through the `<NotebookViewer>` component on the wrapper MDX page. Snapshots are committed so `fern docs dev` works without an API key.

```bash
make generate-fern-notebooks                 # convert existing docs/colab_notebooks/*.ipynb
make generate-fern-notebooks-with-outputs    # full pipeline: execute → colabify → convert (needs NVIDIA_API_KEY)
```

The converter (`fern/scripts/ipynb-to-fern-json.py`) auto-strips the leading Colab badge cell — `<NotebookViewer>` renders its own banner from each wrapper's `colabUrl` prop.

## Local preview

```bash
cd fern
fern docs dev
# → http://localhost:3000
```

If the **Python API** sidebar folder is empty, run `fern docs md generate` (step 1 above) — `fern docs dev` doesn't run library generation itself.

## Versioning

Floating-latest pattern (matches NeMo Curator):

```
fern/versions/
├── latest.yml            ← Unix symlink → v0.5.8.yml
├── v0.5.8.yml            ← real nav file (paths point at ./v0.5.8/pages/...)
└── v0.5.8/pages/...      ← canonical MDX tree
```

`docs.yml` registers both `slug: latest` and `slug: v0.5.8`, so the same MDX renders at `/latest/...` and `/v0.5.8/...`.

### Cutting a new release

```bash
cd fern/versions
cp -R v0.5.8 v0.5.9
cp v0.5.8.yml v0.5.9.yml
sed -i '' 's|./v0.5.8/pages/|./v0.5.9/pages/|g' v0.5.9.yml
ln -sf v0.5.9.yml latest.yml
```

Then add a `v0.5.9` entry to `docs.yml`'s `versions:` list and update the `latest` entry's `display-name`.

## Folder layout

```
fern/
├── README.md                  ← this file
├── docs.yml                   ← title, colors, versions:, libraries:, redirects, custom domain
├── fern.config.json           ← organization, fern-api version pin
├── main.css                   ← bundled NVIDIA theme CSS
├── assets/                    ← logos, favicon, recipe assets, devnote post images
├── images/                    ← /images/* references from MDX (mirror of docs/images)
├── styles/                    ← component-level CSS (notebook-viewer, authors, metrics-table, …)
├── components/                ← React components used by MDX
│   ├── NotebookViewer.tsx     ← renders converted .ipynb cells
│   ├── Authors.tsx            ← devnote bylines (uses devnotes/authors-data.ts)
│   ├── MetricsTable.tsx       ← benchmark tables w/ best-value highlight
│   ├── TrajectoryViewer.tsx   ← multi-turn tool-call traces
│   ├── ExpandableCode.tsx     ← collapsible code (currently unused — Fern SSR has issues)
│   ├── BadgeLinks.tsx, Tag.tsx, CustomCard.tsx, CustomFooter.tsx
│   ├── notebooks/             ← per-tutorial *.json (canonical) + *.ts (MDX import target)
│   └── devnotes/              ← .authors.yml, authors-data.ts, per-post trajectory data
├── scripts/
│   └── ipynb-to-fern-json.py  ← .ipynb → fern/components/notebooks/*.{json,ts}
├── code-reference/            ← gitignored; populated by `fern docs md generate`
└── versions/
    ├── latest.yml -> v0.5.8.yml
    ├── v0.5.8.yml             ← navigation tree
    └── v0.5.8/pages/          ← MDX content
```

## Common commands

| Command | Purpose |
|---------|---------|
| `fern docs dev` | Local preview at `http://localhost:3000` |
| `fern check` | Validate `docs.yml` and MDX |
| `fern docs md generate` | Generate library API docs (no token) |
| `fern generate --docs --preview` | Hosted preview on `*.docs.buildwithfern.com` (needs `FERN_TOKEN`) |
| `make generate-fern-notebooks` | Refresh notebook snapshots from existing colab `.ipynb` |
| `make generate-fern-notebooks-with-outputs` | Full notebook pipeline: execute (needs `NVIDIA_API_KEY`) → colabify → convert |
