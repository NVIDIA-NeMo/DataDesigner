# docs/ vs fern/ Comparison

This document compares the MkDocs `docs/` structure with the Fern `fern/` structure and what needs to be migrated or generated.

## Prerequisites for NotebookViewer

**Run before viewing "The Basics (NotebookViewer)" page:**

```bash
# 1. Generate Colab notebooks (from docs/notebook_source/*.py)
make generate-colab-notebooks

# 2. Convert to Fern format (JSON + TS for NotebookViewer)
make generate-fern-notebooks
```

The NotebookViewer page imports from `@/components/notebooks/1-the-basics` (the `.ts` file). If you haven't run `make generate-fern-notebooks`, that file won't exist and the page will error.

---

## Structure Comparison

| docs/ (MkDocs) | fern/ (Fern) | Status |
|----------------|--------------|--------|
| `index.md` | `v0.5.0/pages/index.mdx` | Migrated |
| `CONTRIBUTING.md` | `v0.5.0/pages/contributing.mdx` | Migrated |
| `concepts/*.md` | `v0.5.0/pages/concepts/*.mdx` | Migrated |
| `recipes/*.md` | `v0.5.0/pages/recipes/*.mdx` | Migrated |
| `plugins/*.md` | `v0.5.0/pages/plugins/*.mdx` | Migrated |
| `code_reference/*.md` | `v0.5.0/pages/api-reference/*.mdx` | Migrated |
| `devnotes/` | Not in fern | Optional – dev notes not migrated |
| `notebook_source/*.py` | N/A | Source only – Jupytext converts to ipynb |
| `colab_notebooks/*.ipynb` | `assets/notebooks/*.json` + `*.ts` | **Generated** by `make generate-fern-notebooks` |
| `assets/recipes/*.py` | `assets/recipes/*.py` | **Copied** – same structure |
| `css/`, `js/`, `overrides/` | `styles/notebook-viewer.css` | Fern uses different theming |

---

## Assets: What's Where

### docs/assets/

| Path | Purpose |
|------|---------|
| `recipes/code_generation/*.py` | Recipe scripts (downloadable) |
| `recipes/qa_and_chat/*.py` | Recipe scripts |
| `recipes/mcp_and_tooluse/*.py` | Recipe scripts |

### fern/assets/

| Path | Purpose |
|------|---------|
| `recipes/` | **Same as docs** – recipe .py files (already migrated) |
| `notebooks/*.ts` | **Generated** – NotebookViewer data (in components tree for import) |
| `favicon.png` | Referenced in docs.yml |

---

## Do You Need to Migrate/Copy?

### Already in fern (no action needed)

- All `assets/recipes/` Python files – same layout as docs
- All MDX pages – migrated from docs
- Favicon, logo – in docs.yml

### Generated (run make)

- `fern/components/notebooks/*.ts` – from `make generate-fern-notebooks`

### Not migrated (optional)

- `docs/devnotes/` – blog-style dev notes
- `docs/css/`, `docs/js/`, `docs/overrides/` – MkDocs-specific

### Recipe download links

Fern recipe pages link to GitHub: `https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/docs/assets/recipes/...`

Those files live in **docs/** in the repo. Fern has a copy in `fern/assets/recipes/` for local reference, but the public download URLs point to `docs/` on GitHub. No change needed unless you want to point to `fern/assets/` instead.

---

## Quick Reference

| Task | Command |
|------|---------|
| Generate Colab notebooks | `make generate-colab-notebooks` |
| Generate Fern notebook JSON/TS | `make generate-fern-notebooks` |
| Preview Fern docs | `fern docs dev` (from project root) |
