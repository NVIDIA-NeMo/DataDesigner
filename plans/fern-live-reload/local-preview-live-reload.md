# Restore Fern fallback live reload

## Context

The unauthenticated Fern fallback added in PR #768 builds a temporary preview
root with local NVIDIA-style theme settings. Its authored content points back to
the repository through symlinks. Fern watches the temporary root, so edits made
under the real `fern/` directory do not produce reload events in the watched
tree.

Serving the real tree without a theme is not an acceptable fallback. Fern may
render it differently across CLI releases, and it does not guarantee a close
match to the published NVIDIA theme.

## Goal

Keep the deterministic local NVIDIA-style fallback while making edits under
`fern/` reliably trigger Fern live reload.

## Design

Materialize the fallback as real files in the temporary preview root, then poll
the source tree for metadata changes and mirror only changed paths while the
Fern child process is running.

- Keep the authenticated path in `Makefile` unchanged.
- Keep the existing local theme values and `styles/local-preview.css` unchanged.
- Copy the initial `fern/` tree into the temporary root instead of creating
  symlinks to the source tree. The current tree is about 19 MB.
- Track source files by relative path, type, modification time, inode, and size.
  Poll every 250 ms without adding a file-watching dependency.
- Copy additions and changes into the temporary root and remove deleted paths.
  Fern observes those writes inside its watched tree and reloads normally.
- Regenerate the transformed `docs.yml` whenever the source `docs.yml` changes.
  Continue removing `global-theme` and applying the deterministic local theme.
- Preserve extensionless component aliases, but materialize them inside the
  temporary tree instead of linking them to source files.
- Retry files that disappear during an editor save on the next pass. Stop the
  child and exit nonzero for other synchronization failures rather than
  continuing with a stale preview.

This avoids mutating the working tree, keeps source files editable at their
normal paths, and works without Fern support for an alternate config file.

## Tasks

- [x] Add focused tests under `fern/scripts/tests/` for the preview wrapper.
  Cover initial materialization, page edits, file additions and deletions,
  `docs.yml` regeneration, and component alias updates. Assert that preview
  files do not resolve outside the temporary root.
- [x] Update `fern/scripts/serve-local-docs-preview.py` to build a real preview
  tree and expose one synchronization pass that compares source snapshots and
  applies changes. Handle files disappearing during editor atomic-save
  sequences by retrying on the next pass.
- [x] Change child-process orchestration to synchronize every 250 ms until Fern
  exits. Preserve interrupt forwarding and ensure synchronization errors stop
  the child cleanly.
- [x] Update `fern/README.md` troubleshooting notes to state that the fallback
  is deterministic and supports live reload without Fern theme access.
- [x] Validate with:
  - `.venv/bin/pytest fern/scripts/tests/test_serve_local_docs_preview.py`
  - `.venv/bin/ruff check --fix .`
  - `.venv/bin/ruff format .`
  - `make check-fern-docs`
  - `make serve-fern-docs-local-theme`, followed by editing an MDX page, a TSX
    component, and `docs.yml` and confirming each change reloads in the browser

## Open Questions

None. The polling interval and full initial copy are intentionally simple for a
small local-only documentation tree.
