#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

compute_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

create_snapshot() {
    local root="$1"
    local release_tag="$2"
    local mode="$3"
    local archive="$4"
    local asset_name
    local digest

    if [ "$mode" != "executed" ] && [ "$mode" != "source-fallback" ]; then
        echo "Invalid notebook snapshot mode: $mode" >&2
        exit 1
    fi
    if [ ! -d "$root/fern/components/notebooks" ]; then
        echo "Missing rendered notebooks: $root/fern/components/notebooks" >&2
        exit 1
    fi

    asset_name="$(basename "$archive")"
    mkdir -p "$(dirname "$archive")"
    tar -czf "$archive" -C "$root" fern/components/notebooks
    digest="$(compute_sha256 "$archive")"

    jq -n \
        --arg release_tag "$release_tag" \
        --arg asset "$asset_name" \
        --arg sha256 "$digest" \
        --arg mode "$mode" \
        --arg run_id "${GITHUB_RUN_ID:-local}" \
        --arg run_attempt "${GITHUB_RUN_ATTEMPT:-1}" \
        '{
          schema_version: 1,
          release_tag: $release_tag,
          asset: $asset,
          sha256: $sha256,
          mode: $mode,
          run_id: $run_id,
          run_attempt: $run_attempt
        }' > "$root/fern/notebook-snapshot.json"
}

restore_snapshot() {
    local root="$1"
    local archive="$2"
    local metadata="$root/fern/notebook-snapshot.json"
    local expected_asset
    local expected_digest
    local actual_digest

    if [ ! -f "$metadata" ]; then
        echo "Missing notebook snapshot metadata: $metadata" >&2
        exit 1
    fi

    expected_asset="$(jq -er '.asset' "$metadata")"
    expected_digest="$(jq -er '.sha256' "$metadata")"
    if [ "$(basename "$archive")" != "$expected_asset" ]; then
        echo "Notebook snapshot asset mismatch: expected $expected_asset" >&2
        exit 1
    fi

    actual_digest="$(compute_sha256 "$archive")"
    if [ "$actual_digest" != "$expected_digest" ]; then
        echo "Notebook snapshot checksum mismatch" >&2
        exit 1
    fi
    if tar -tzf "$archive" | grep -Ev '^fern/components/notebooks(/.*)?$' >/dev/null; then
        echo "Notebook snapshot contains unexpected paths" >&2
        exit 1
    fi

    rm -rf "$root/fern/components/notebooks"
    tar -xzf "$archive" -C "$root"
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 create ROOT RELEASE_TAG MODE ARCHIVE | restore ROOT ARCHIVE" >&2
    exit 1
fi

command="$1"
shift
case "$command" in
    create)
        if [ "$#" -ne 4 ]; then
            echo "Usage: $0 create ROOT RELEASE_TAG MODE ARCHIVE" >&2
            exit 1
        fi
        create_snapshot "$@"
        ;;
    restore)
        if [ "$#" -ne 2 ]; then
            echo "Usage: $0 restore ROOT ARCHIVE" >&2
            exit 1
        fi
        restore_snapshot "$@"
        ;;
    *)
        echo "Unknown command: $command" >&2
        exit 1
        ;;
esac
