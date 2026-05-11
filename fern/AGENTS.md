# Fern Docs Notes

This folder contains the Fern docs site for NeMo Data Designer. Use `fern/README.md` as the detailed guide.

## Publishing Safety

- `make serve-fern-docs-locally` is local-only.
- `make check-fern-docs` is local/CI validation only.
- `fern generate --docs` publishes.
- `fern generate --docs --preview` publishes a hosted preview.
- Do not run publish or preview commands unless the user explicitly asks.

## Generated Artifacts

- `make generate-fern-api-reference` creates gitignored API reference files in `fern/code-reference/`.
- `make generate-fern-notebooks` creates gitignored notebook files in `fern/components/notebooks/`.
- `docs/notebook_source/*.py` is the notebook source of truth.
- `docs/colab_notebooks/` is only for Colab links, not Fern input.

## Versioning Model

Use hybrid versioning. A version YAML may reuse older page files for unchanged content, and copy only changed/new pages into that version's `pages/` tree.

Example: a `v0.5.9` nav entry can point to `./v0.5.8/pages/concepts/columns.mdx`. Users still see the page under the active `v0.5.9` URL because Fern routes by version slug and nav title, not by source file path.

Do not call a version frozen if its YAML points at shared pages that may change later. If a page's content must remain release-specific, copy it into `fern/versions/vX.Y.Z/pages/...` and point that version's YAML to the copy.

## Release Prep

For a future Fern-native release:

1. Add `vX.Y.Z` to `fern/docs.yml`.
2. Add `fern/versions/vX.Y.Z.yml`.
3. Reuse older paths for unchanged pages.
4. Copy only changed/new pages into `fern/versions/vX.Y.Z/pages/...`.
5. Update only those nav paths to the copied pages.
6. Update `latest.yml` and the `Latest · vX.Y.Z` label.
7. Run `make check-fern-docs`.

Older releases before the Fern migration stay on the MkDocs archive through the "Older versions" page and redirects in `docs.yml`.
