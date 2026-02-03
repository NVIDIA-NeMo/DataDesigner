# Versioning Guide

DataDesigner uses **semantic versioning** with automated version management via `uv-dynamic-versioning`.

## How It Works

Versions are automatically derived from git tags:

- **No tag**: `0.1.0.dev<N>+g<commit-hash>` (development version)
- **Tagged commit**: `1.2.3` (release version)
- **After tag**: `1.2.4.dev<N>+g<commit-hash>` (next development version)

## Version Format

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

## Creating a Release

When ready to release version `X.Y.Z`:

```bash
# Tag the release
git tag vX.Y.Z

# Push the tag
git push origin vX.Y.Z

# Build and publish
uv build
uv publish
```

Example:
```bash
git tag v0.1.0
git push origin v0.1.0
```

## Accessing Version in Code

Users can access the version using Python's standard `importlib.metadata`:

```python
import importlib.metadata

print(importlib.metadata.version("data-designer"))
# Output: 0.1.0 (or 0.1.0.dev18+ga7496d01a if between releases)
```

Note: `data_designer.__version__` does not work because `data_designer` is a namespace package.

## Technical Details

- Version source: Git tags via `uv-dynamic-versioning`
- Version access: `importlib.metadata.version("data-designer")` (standard Python approach)
- Configuration: Package `pyproject.toml` files

## For Collaborators

When you clone the repository and run `uv sync`, you can access the version immediately:

```bash
git clone <repo>
uv sync
uv run python -c "import importlib.metadata; print(importlib.metadata.version('data-designer'))"
# Works!
```

## Development Workflow

1. **During development**: Commit normally, version auto-increments as dev versions
2. **Ready to release**: Create and push a git tag (e.g., `v0.1.0`)
3. **After release**: Continue development, version becomes next dev version (e.g., `0.1.1.dev1`)

No manual version bumping required!
