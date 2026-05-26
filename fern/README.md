# Fern Docs

This folder is the Fern Docs build for NeMo Data Designer. The site currently deploys to **`datadesigner.docs.buildwithfern.com/nemo/datadesigner`**; [`docs.yml`](docs.yml) also declares the future `docs.nvidia.com/nemo/datadesigner` custom domain.

## Migration phase

Data Designer is moving from MkDocs to Fern over several releases. During that transition:

- Keep the MkDocs build and release archive working.
- Keep Fern working in parallel for local checks and hosted validation.
- Treat `docs/` as the docs source of truth unless a page has already been intentionally moved to Fern-only MDX.
- Treat `docs/notebook_source/*.py` as the notebook source of truth.
- Keep generated Fern notebook artifacts gitignored.

## Prerequisites

```bash
# Install Fern CLI globally
npm install -g fern-api
```

## First-time setup

One pre-render step is needed before the dev server has all tutorial content. It produces gitignored files and is safe to rerun.

### Notebook tutorials (gitignored - regenerate on clone)

Each tutorial source file is converted to a JSON+TS pair in `fern/components/notebooks/`, then rendered through the `<NotebookViewer>` component on the wrapper MDX page. Output is gitignored; regenerate it after cloning and after changing `docs/notebook_source/*.py`.

```bash
make generate-fern-notebooks                 # convert docs/notebook_source/*.py, preferring docs/notebooks/*.ipynb when present
make generate-fern-notebooks-with-outputs    # full pipeline: execute → colabify → convert (needs NVIDIA_API_KEY)
```

The docs build does not use `docs/colab_notebooks/`; those files exist for the wrapper pages' `colabUrl` links. The converter (`fern/scripts/ipynb-to-fern-json.py`) still strips Colab-only setup cells defensively if run on a Colab notebook.

Fern does not run this conversion automatically. Run `make prepare-fern-docs` before local preview/checks, and run the same notebook conversion in CI before `fern generate --docs`.

## Local preview

```bash
make serve-fern-docs-locally
# → http://localhost:3000
```

`serve-fern-docs-locally` generates notebook artifacts before starting `fern docs dev`. It does not publish.

## CI and publishing

Fern publishing runs alongside MkDocs during migration:

- `.github/workflows/build-fern-docs.yml` runs on release publication or manual dispatch. It snapshots release docs into the CI-managed `docs-website` branch, builds executed notebooks from the release source, runs `make check-fern-docs` from `docs-website`, and publishes Fern.
- `.github/workflows/publish-fern-devnotes.yml` runs on `main` when Dev Notes or Fern Dev Notes assets change, plus manual dispatch. It patches only Dev Notes into the `docs-website` branch's current latest docs, reuses the last docs notebook artifact, runs `make check-fern-docs`, and publishes Fern.
- `.github/workflows/docs-preview.yml` remains the PR preview workflow and posts both MkDocs and Fern preview links for same-repository PRs. It converts tutorial sources without execution outputs for preview builds. Fork PRs still run docs build/checks, but skip hosted previews because those require deployment secrets.

These workflows require the org-level `DOCS_FERN_TOKEN` secret. The workflows expose it to the Fern CLI as `FERN_TOKEN`.

Fern release snapshots live on `docs-website`, not on `main`. This mirrors the MkDocs `gh-pages` model without mixing Fern source state into the MkDocs output branch. The branch stores a source snapshot, not only `fern/`, because `make check-fern-docs` needs the Python packages and workspace metadata. Pushes to `docs-website` use `GITHUB_TOKEN`, so publishing happens inline in the same workflow instead of relying on a second workflow trigger.

The `docs-website` branch is an orphan-style publish branch. Published commits include `fern/publish-metadata.json` with the source repository, ref, SHA, release tag when applicable, and published branch.

The `docs-website` branch must already contain the historical Fern archive (`v0.6.0`, `v0.5.9`, `v0.5.8`, and `older`) before release publishing runs. The workflow fails if those redirect targets are missing.

Manual dispatch with `release_tag` creates or refreshes that release snapshot. For the already-published `v0.6.0` release, run **Build Fern docs** with `release_tag=v0.6.0` and `source_ref=main` after this fix merges. Future release events default `source_ref` to the release tag.

## Versioning

`main` contains only the latest Fern authoring docs. Published release snapshots live on `docs-website`.

```
fern/versions/
├── latest.yml          ← authoring nav
└── latest/pages/...    ← authoring pages
```

The CI-managed `docs-website` branch has the published archive:

```
fern/versions/
├── latest.yml
├── latest/pages/...
├── v0.6.0.yml
├── v0.6.0/pages/...
├── v0.5.9.yml
├── v0.5.9/pages/...
├── v0.5.8.yml
├── v0.5.8/pages/...
├── older.yml
└── older/pages/...
```

Each frozen `vX.Y.Z.yml` nav on `docs-website` must point only at that version's own `vX.Y.Z/pages/...` files. The release sync materializes shared historical pages into each version folder before publishing.

Normal GitHub releases do not need a dedicated pre-release Fern PR. The release workflow snapshots the release into `docs-website` and publishes from that branch.

Dev Notes publishing mirrors MkDocs: it patches only the Dev Notes nav and pages from `main` into the current latest docs on `docs-website`, then republishes Fern.

## Folder layout

```
fern/
├── README.md                  ← this file
├── docs.yml                   ← title, colors, versions:, redirects, custom domain
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
│   ├── notebooks/             ← gitignored per-tutorial *.json + *.ts output
│   └── devnotes/              ← .authors.yml, authors-data.ts, per-post trajectory data
├── scripts/
│   └── ipynb-to-fern-json.py  ← .ipynb → fern/components/notebooks/*.{json,ts}
└── versions/
    ├── latest.yml             ← authoring navigation tree
    └── latest/pages/          ← authoring MDX content
```

## Common commands

Primary local commands:

| Command | Purpose |
|---------|---------|
| `make check-fern-docs-locally` | Install docs dependencies, generate Fern artifacts, and run `fern check` |
| `make serve-fern-docs-locally` | Generate local Fern artifacts and serve local docs |
| `make generate-fern-notebooks-with-outputs` | Full notebook pipeline: execute (needs `NVIDIA_API_KEY`) → colabify → convert |
| `make prepare-fern-release VERSION=X.Y.Z` | Add or refresh Fern version files for release preview |
| `make check-fern-release-version VERSION=X.Y.Z REQUIRE_LATEST=1` | Verify Fern release metadata exists before publishing |

Support and CI targets:

| Command | Purpose |
|---------|---------|
| `make install-docs-deps` | Install docs and notebook dependencies |
| `make generate-fern-notebooks` | Refresh gitignored notebook output from `docs/notebook_source/*.py` |
| `make prepare-fern-docs` | Generate local Fern notebook artifacts |
| `make check-fern-docs` | Generate local Fern notebook artifacts and run `fern check` |

Raw Fern CLI commands, normally wrapped by Make:

| Command | Purpose |
|---------|---------|
| `fern docs dev` | Local preview at `http://localhost:3000` |
| `fern check` | Validate `docs.yml` and MDX |
| `fern generate --docs --preview` | Hosted preview on `*.docs.buildwithfern.com` (needs Fern token) |
