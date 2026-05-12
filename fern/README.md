# Fern Docs

This folder is the Fern Docs build for NeMo Data Designer. The site currently deploys to **`datadesigner.docs.buildwithfern.com/nemo/datadesigner`**; [`docs.yml`](docs.yml) also declares the future `docs.nvidia.com/nemo/datadesigner` custom domain.

## Migration phase

Data Designer is moving from MkDocs to Fern over several releases. During that transition:

- Keep the MkDocs build and release archive working.
- Keep Fern working in parallel for local checks and hosted validation.
- Treat `docs/` as the docs source of truth unless a page has already been intentionally moved to Fern-only MDX.
- Treat `docs/notebook_source/*.py` as the notebook source of truth.
- Keep generated Fern API reference and notebook artifacts gitignored.

## Prerequisites

```bash
# Install Fern CLI globally
npm install -g fern-api
```

## First-time setup

Two pre-render steps are needed before the dev server has all content. Both produce gitignored files and are safe to rerun.

### 1. Python API reference (gitignored - must regenerate)

`make generate-fern-api-reference` uses `py2fern` to extract API docs from the local Python source (`packages/data-designer-config/src/data_designer/config`). The output lands in `fern/code-reference/data-designer/` (gitignored).

```bash
make generate-fern-api-reference
```

The `libraries:` block in [`docs.yml`](docs.yml) still documents the equivalent Fern-native generator. Run `make generate-fern-api-reference-native` only when you want the Fern CLI output and have Fern auth.

Re-run when the upstream package source changes.

### 2. Notebook tutorials (gitignored - regenerate on clone)

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

`serve-fern-docs-locally` generates Fern API reference and notebook artifacts before starting `fern docs dev`. It does not publish.

## CI and publishing

Fern publishing runs alongside MkDocs during migration:

- `.github/workflows/build-fern-docs.yml` runs on release publication or manual dispatch. It builds executed notebooks, runs `make check-fern-docs`, and publishes Fern.
- `.github/workflows/publish-fern-devnotes.yml` runs on `main` when Dev Notes or Fern Dev Notes assets change, plus manual dispatch. It reuses the last docs notebook artifact, runs `make check-fern-docs`, and publishes Fern.
- `.github/workflows/docs-preview.yml` remains the PR preview workflow and posts both MkDocs and Fern preview links for same-repository PRs. It converts tutorial sources without execution outputs for preview builds. Fork PRs still run docs build/checks, but skip hosted previews because those require deployment secrets.

These workflows require the org-level `DOCS_FERN_TOKEN` secret. The workflows expose it to the Fern CLI as `FERN_TOKEN`.

Release publishing also runs `fern/scripts/fern-release-version.py check` before building notebooks. A release fails early if the release tag is not represented in `docs.yml` and `versions/vX.Y.Z.yml`. Manual dispatch can validate a specific tag through the workflow's `release_tag` input; otherwise it uses the latest published release.

## Versioning

Current Fern versions:

```
fern/versions/
├── latest.yml          ← rolling nav file
├── latest/pages/...    ← latest-only page overrides
├── v0.5.9.yml          ← release nav file
├── v0.5.9/pages/...    ← v0.5.9-only page overrides
├── v0.5.8.yml          ← first migrated release nav file
├── v0.5.8/pages/...    ← shared migrated MDX tree
├── older.yml           ← older versions landing page
└── older/pages/...     ← links to the MkDocs archive
```

`docs.yml` registers `slug: latest`, `slug: v0.5.9`, `slug: v0.5.8`, and `slug: older-versions`. The `latest` and `v0.5.9` nav files intentionally reuse the migrated `v0.5.8/pages/` tree for most content so the first Fern-native versions do not duplicate every page. Add version-specific page copies only when content diverges.

Fern version URLs are based on the active version entry, not the source file path. A `v0.5.9` page can point at `./v0.5.8/pages/...` and still render under `/nemo/datadesigner/v0.5.9/...`; users do not see the reused source path.

Dev Notes are versioned: `latest.yml` can include posts from `main` that are not in old release navs yet. Frozen release navs (`v0.5.9.yml`, `v0.5.8.yml`) should include only posts available at that release point.

Released versions older than `v0.5.8` stay on the MkDocs archive at `https://nvidia-nemo.github.io/DataDesigner/<version>/`. The Fern version picker includes an "Older versions" page linking to those archives.

When cutting future Fern-native versions, use a hybrid model:

1. Run `make prepare-fern-release VERSION=X.Y.Z`.
2. Review the generated `docs.yml` and `versions/vX.Y.Z.yml` changes.
3. Reuse older page paths for unchanged content.
4. Copy changed/new pages into `versions/vX.Y.Z/pages/...`.
5. Point only changed nav entries at the copied `vX.Y.Z` pages.
6. Update `latest.yml` if the rolling docs should diverge after release prep.
7. Run `make check-fern-docs`.

Do not add a new version entry without deciding which pages are release-specific. A version YAML that points at shared pages can drift when those shared files change later.

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
│   ├── notebooks/             ← gitignored per-tutorial *.json + *.ts output
│   └── devnotes/              ← .authors.yml, authors-data.ts, per-post trajectory data
├── scripts/
│   └── ipynb-to-fern-json.py  ← .ipynb → fern/components/notebooks/*.{json,ts}
├── code-reference/            ← gitignored; populated by `make generate-fern-api-reference`
└── versions/
    ├── latest.yml             ← rolling navigation tree
    ├── v0.5.9.yml             ← release navigation tree
    ├── v0.5.8.yml             ← navigation tree
    ├── older.yml              ← older versions landing page
    └── v0.5.8/pages/          ← shared MDX content
```

## Common commands

Primary local commands:

| Command | Purpose |
|---------|---------|
| `make check-fern-docs-locally` | Install docs dependencies, generate Fern artifacts, and run `fern check` |
| `make serve-fern-docs-locally` | Generate local Fern artifacts and serve local docs |
| `make generate-fern-notebooks-with-outputs` | Full notebook pipeline: execute (needs `NVIDIA_API_KEY`) → colabify → convert |
| `make prepare-fern-release VERSION=X.Y.Z` | Add Fern version files before cutting a release |
| `make check-fern-release-version VERSION=X.Y.Z` | Verify Fern release metadata exists before publishing |

Support and CI targets:

| Command | Purpose |
|---------|---------|
| `make install-docs-deps` | Install docs and notebook dependencies |
| `make generate-fern-api-reference` | Generate local Fern API reference with `py2fern` |
| `make generate-fern-api-reference-native` | Generate Fern API reference with Fern CLI (requires Fern auth) |
| `make generate-fern-notebooks` | Refresh gitignored notebook output from `docs/notebook_source/*.py` |
| `make prepare-fern-docs` | Generate local Fern artifacts |
| `make check-fern-docs` | Generate local Fern artifacts and run `fern check` |

Raw Fern CLI commands, normally wrapped by Make:

| Command | Purpose |
|---------|---------|
| `fern docs dev` | Local preview at `http://localhost:3000` |
| `fern check` | Validate `docs.yml` and MDX |
| `fern docs md generate` | Generate library API docs with Fern CLI (requires Fern auth) |
| `fern generate --docs --preview` | Hosted preview on `*.docs.buildwithfern.com` (needs Fern token) |
