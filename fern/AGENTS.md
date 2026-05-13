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

Before editing a file under an older shared tree such as `fern/versions/v0.5.8/pages/...`, check every `fern/versions/*.yml` file that points at it. If the content describes a newer release, copy it into the target `vX.Y.Z/pages/...` tree and retarget that release. Point `latest.yml` at the released copy unless latest has already diverged after that release.

Dev Notes are versioned. Do not add a new release-specific post to an older frozen nav or page tree. If a post says "As of Data Designer vX.Y.Z", it belongs in `latest.yml` and `vX.Y.Z.yml`, not in older version navs.

Do not call a version frozen if its YAML points at shared pages that may change later. If a page's content must remain release-specific, copy it into `fern/versions/vX.Y.Z/pages/...` and point that version's YAML to the copy.

## Release Prep

Normal GitHub releases do not need a dedicated pre-release Fern PR. The release workflow snapshots Fern docs into the CI-managed `docs-website` branch and publishes from that branch.

If a user asks to preview or hand-curate the release docs before tagging:

1. Run `make prepare-fern-release VERSION=X.Y.Z`.
2. Review the generated `fern/docs.yml` and `fern/versions/vX.Y.Z.yml` changes.
3. Reuse older paths for unchanged pages.
4. Copy only changed/new pages into `fern/versions/vX.Y.Z/pages/...`.
5. Update only those nav paths to the copied pages.
6. Update `latest.yml` if the rolling docs should diverge after release prep.
7. Run `make check-fern-docs`.

Release publishing runs `fern/scripts/fern-published-branch.py sync-source`, then `fern/scripts/fern-release-version.py prepare --force` and `check`. If `latest.yml` cannot be made to match the release nav on `docs-website`, the workflow should fail early.

Older releases before the Fern migration stay on the MkDocs archive through the "Older versions" page and redirects in `docs.yml`.
