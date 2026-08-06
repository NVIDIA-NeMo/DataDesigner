#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build notebooks with per-file caching. Only re-executes notebooks whose
# source or runtime context changed since the last cached build.
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
DOCS_JUPYTEXT="${DOCS_JUPYTEXT:-$REPO_ROOT/.venv/bin/jupytext}"
NOTEBOOK_EXECUTION_ATTEMPTS="${NOTEBOOK_EXECUTION_ATTEMPTS:-1}"
NOTEBOOK_RETRY_DELAY_SECONDS="${NOTEBOOK_RETRY_DELAY_SECONDS:-15}"

if [[ ! "$NOTEBOOK_EXECUTION_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ NOTEBOOK_EXECUTION_ATTEMPTS must be a positive integer"
    exit 1
fi
if [[ ! "$NOTEBOOK_RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "❌ NOTEBOOK_RETRY_DELAY_SECONDS must be a non-negative integer"
    exit 1
fi

if [ ! -x "$DOCS_JUPYTEXT" ]; then
    echo "❌ Missing jupytext executable: $DOCS_JUPYTEXT"
    echo "Run 'make install-dev-notebooks' first."
    exit 1
fi

execute_notebook() {
    local src="$1"
    local output="${src%.py}.ipynb"
    local attempt
    local delay

    for ((attempt = 1; attempt <= NOTEBOOK_EXECUTION_ATTEMPTS; attempt++)); do
        rm -f "$output"
        if "$DOCS_JUPYTEXT" --to ipynb --execute "$src"; then
            return 0
        fi
        rm -f "$output"
        if [ "$attempt" -eq "$NOTEBOOK_EXECUTION_ATTEMPTS" ]; then
            return 1
        fi
        delay=$((NOTEBOOK_RETRY_DELAY_SECONDS * attempt))
        echo "  ⚠️  Attempt $attempt failed; retrying in ${delay}s"
        sleep "$delay"
    done
}

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Copy static files
cp "$SOURCE_DIR/_README.md" "$OUTPUT_DIR/README.md"
cp "$SOURCE_DIR/_pyproject.toml" "$OUTPUT_DIR/pyproject.toml"

needs_cleanup=false

for src in "$SOURCE_DIR"/*.py; do
    name="$(basename "$src" .py)"
    hash="$(compute_sha256 "$src"):${NOTEBOOK_CACHE_CONTEXT:-}"
    cached_hash_file="$CACHE_DIR/${name}.sha256"
    cached_notebook="$CACHE_DIR/${name}.ipynb"

    if [ -f "$cached_hash_file" ] && [ -f "$cached_notebook" ] && [ "$(cat "$cached_hash_file")" = "$hash" ]; then
        echo "  ✅ $name.ipynb - cached (unchanged)"
        cp "$cached_notebook" "$OUTPUT_DIR/${name}.ipynb"
    else
        echo "  🔄 $name.ipynb - executing..."
        execute_notebook "$src"
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
