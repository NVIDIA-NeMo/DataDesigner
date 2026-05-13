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

`main` contains only the latest Fern authoring docs under `fern/versions/latest.yml` and `fern/versions/latest/pages/...`.

Published release snapshots live on the CI-managed `docs-website` branch. Do not manually edit `docs-website` unless the user explicitly asks for release archive repair.

The `docs-website` branch must already contain the historical Fern archive (`v0.6.0`, `v0.5.9`, `v0.5.8`, and `older`). The release workflow fails if those redirect targets are missing.

Dev Notes publishing patches only Dev Notes from `main` into the current latest docs on `docs-website`, mirroring the MkDocs workflow.

## Release Prep

Normal GitHub releases do not need a dedicated pre-release Fern PR. The release workflow snapshots Fern docs into the CI-managed `docs-website` branch and publishes from that branch.

Release publishing runs `fern/scripts/fern-published-branch.py sync-source`, then `fern/scripts/fern-release-version.py prepare --force` and `check`. If `latest.yml` cannot be made to match the release nav on `docs-website`, the workflow should fail early.

Older releases before the Fern migration stay on the MkDocs archive through the "Older versions" page and redirects in `docs.yml`.
