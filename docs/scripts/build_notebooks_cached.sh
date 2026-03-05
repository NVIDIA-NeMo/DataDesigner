#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build notebooks with per-file caching. Only re-executes notebooks whose
# source .py file changed since the last cached build.
#
# Usage:
#   ./docs/scripts/build_notebooks_cached.sh [CACHE_DIR]
#
# CACHE_DIR defaults to .notebook-cache

set -euo pipefail

compute_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_DIR="$REPO_ROOT/docs/notebook_source"
OUTPUT_DIR="$REPO_ROOT/docs/notebooks"
CACHE_DIR="${1:-$REPO_ROOT/.notebook-cache}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Copy static files
cp "$SOURCE_DIR/_README.md" "$OUTPUT_DIR/README.md"
cp "$SOURCE_DIR/_pyproject.toml" "$OUTPUT_DIR/pyproject.toml"

needs_cleanup=false

for src in "$SOURCE_DIR"/*.py; do
    name="$(basename "$src" .py)"
    hash="$(compute_sha256 "$src")"
    cached_hash_file="$CACHE_DIR/${name}.sha256"
    cached_notebook="$CACHE_DIR/${name}.ipynb"

    if [ -f "$cached_hash_file" ] && [ -f "$cached_notebook" ] && [ "$(cat "$cached_hash_file")" = "$hash" ]; then
        echo "  ✅ $name.ipynb — cached (unchanged)"
        cp "$cached_notebook" "$OUTPUT_DIR/${name}.ipynb"
    else
        echo "  🔄 $name.ipynb — executing..."
        uv run --all-packages --group notebooks --group docs jupytext --to ipynb --execute "$src"
        mv "$SOURCE_DIR/${name}.ipynb" "$OUTPUT_DIR/${name}.ipynb"
        needs_cleanup=true

        # Update cache
        cp "$OUTPUT_DIR/${name}.ipynb" "$cached_notebook"
        echo "$hash" > "$cached_hash_file"
    fi
done

if [ "$needs_cleanup" = true ]; then
    # Clean up artifacts from executed notebooks
    [ -d "$SOURCE_DIR/artifacts" ] && rm -rf "$SOURCE_DIR/artifacts"
    find "$SOURCE_DIR" -name '*.csv' -delete 2>/dev/null || true
fi

echo "✅ Notebooks ready in $OUTPUT_DIR"
