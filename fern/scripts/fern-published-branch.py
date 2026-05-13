#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync Fern authoring content into the CI-managed publish branch."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

DEVNOTES_SECTION_RE = re.compile(r"^  - section:\s+Dev Notes\s*$")
NAV_PATH_RE = re.compile(r"^(\s*path:\s+)\./([^#\s]+)(.*)$")
REDIRECT_VERSION_RE = re.compile(
    r'^\s*destination:\s+["\']/nemo/datadesigner/((?:v[0-9][^/"\']*)|older-versions)(?:/|["\'])'
)
VERSION_SLUG_RE = re.compile(r"^\s*slug:\s+['\"]?([^'\"\s]+)")
SKIP_NAMES = {
    ".git",
    ".mypy_cache",
    ".notebook-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "dist",
    "site",
}
FERN_DEVNOTE_SUPPORT_PATHS = [
    "fern/assets/devnotes",
    "fern/components/Authors.tsx",
    "fern/components/BlogCard.tsx",
    "fern/components/devnotes",
    "fern/styles/authors.css",
    "fern/styles/blog-card.css",
]


class PublishedBranchError(RuntimeError):
    pass


def find_top_level_block(lines: list[str], name: str) -> tuple[int, int]:
    start = next((i for i, line in enumerate(lines) if line == f"{name}:\n"), -1)
    if start == -1:
        raise PublishedBranchError(f"Missing top-level '{name}:' block")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z0-9_-]+:", lines[i]):
            end = i
            break
    return start, end


def versions_block(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    lines = path.read_text().splitlines(keepends=True)
    try:
        start, end = find_top_level_block(lines, "versions")
    except PublishedBranchError:
        return None
    return lines[start:end]


def restore_versions_block(path: Path, block: list[str] | None) -> None:
    if block is None:
        return
    lines = path.read_text().splitlines(keepends=True)
    start, end = find_top_level_block(lines, "versions")
    lines[start:end] = block
    path.write_text("".join(lines))


def required_redirect_slugs(path: Path) -> set[str]:
    required: set[str] = set()
    for line in path.read_text().splitlines():
        match = REDIRECT_VERSION_RE.match(line)
        if match:
            required.add(match.group(1))
    return required


def version_slugs(path: Path) -> set[str]:
    slugs: set[str] = set()
    for line in versions_block(path) or []:
        match = VERSION_SLUG_RE.match(line)
        if match:
            slugs.add(match.group(1))
    return slugs


def validate_redirect_targets(published_root: Path) -> None:
    docs_yml = published_root / "fern" / "docs.yml"
    missing = sorted(required_redirect_slugs(docs_yml) - version_slugs(docs_yml))
    if missing:
        formatted = ", ".join(missing)
        raise PublishedBranchError(
            f"Published Fern docs.yml is missing version entries required by redirects: {formatted}. "
            "Initialize docs-website with the historical Fern archive before publishing."
        )


def ignore_source(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_NAMES or name == "__pycache__"}


def copy_path(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, ignore=ignore_source)
    else:
        shutil.copy2(source, target)


def clear_published_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for path in root.iterdir():
        if path.name == ".git":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def merge_preserved_versions(source_versions: Path, published_versions: Path, preserved_versions: Path) -> None:
    if not preserved_versions.exists():
        return
    published_versions.mkdir(parents=True, exist_ok=True)
    for path in preserved_versions.iterdir():
        target = published_versions / path.name
        source_peer = source_versions / path.name
        if source_peer.exists():
            continue
        copy_path(path, target)


def sync_source(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root)
    published_root = Path(args.published_root)
    if not (source_root / "fern" / "docs.yml").exists():
        raise PublishedBranchError(f"Missing source Fern docs at {source_root / 'fern'}")

    preserved_versions_block = versions_block(published_root / "fern" / "docs.yml")
    with tempfile.TemporaryDirectory() as tmpdir:
        preserved_versions = Path(tmpdir) / "versions"
        if (published_root / "fern" / "versions").exists():
            shutil.copytree(published_root / "fern" / "versions", preserved_versions)

        clear_published_tree(published_root)
        shutil.copytree(source_root, published_root, dirs_exist_ok=True, ignore=ignore_source)
        merge_preserved_versions(
            source_root / "fern" / "versions", published_root / "fern" / "versions", preserved_versions
        )
        restore_versions_block(published_root / "fern" / "docs.yml", preserved_versions_block)
        validate_redirect_targets(published_root)
    return 0


def extract_devnotes_block(path: Path) -> list[str]:
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if DEVNOTES_SECTION_RE.match(line)), -1)
    if start == -1:
        raise PublishedBranchError(f"Dev Notes section not found in {path}")
    end = start + 1
    while end < len(lines):
        if lines[end].startswith("  - ") and lines[end].strip():
            break
        end += 1
    return lines[start:end]


def rewrite_devnotes_block(source_root: Path, published_root: Path, block: list[str]) -> list[str]:
    rewritten: list[str] = []
    for line in block:
        match = NAV_PATH_RE.match(line)
        if not match:
            rewritten.append(line)
            continue
        rel_path = Path(match.group(2))
        if "pages/devnotes" not in rel_path.as_posix():
            rewritten.append(line)
            continue
        source_file = source_root / "fern" / "versions" / rel_path
        if not source_file.exists():
            raise PublishedBranchError(
                f"Missing Dev Notes page referenced by {source_root / 'fern' / 'versions'}: {rel_path}"
            )
        target_rel = Path("latest/pages/devnotes") / rel_path.as_posix().split("pages/devnotes/", 1)[1]
        target_file = published_root / "fern" / "versions" / target_rel
        copy_path(source_file, target_file)
        rewritten.append(f"{match.group(1)}./{target_rel.as_posix()}{match.group(3)}\n")
    return rewritten


def replace_devnotes_block(path: Path, block: list[str]) -> None:
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if DEVNOTES_SECTION_RE.match(line)), -1)
    if start == -1:
        raise PublishedBranchError(f"Dev Notes section not found in {path}")
    end = start + 1
    while end < len(lines):
        if lines[end].startswith("  - ") and lines[end].strip():
            break
        end += 1
    lines[start:end] = block
    path.write_text("".join(lines))


def patch_devnotes(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root)
    published_root = Path(args.published_root)
    source_nav = source_root / "fern" / "versions" / "latest.yml"
    target_nav = published_root / "fern" / "versions" / "latest.yml"
    if not source_nav.exists():
        raise PublishedBranchError(f"Missing {source_nav}")
    if not target_nav.exists():
        raise PublishedBranchError(f"Missing {target_nav}; publish a Fern release snapshot first")

    for rel_path in FERN_DEVNOTE_SUPPORT_PATHS:
        copy_path(source_root / rel_path, published_root / rel_path)

    source_block = extract_devnotes_block(source_nav)
    replace_devnotes_block(target_nav, rewrite_devnotes_block(source_root, published_root, source_block))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    sync_parser = subparsers.add_parser("sync-source")
    sync_parser.add_argument("--source-root", required=True, help="Repository checkout with authoring content")
    sync_parser.add_argument("--published-root", required=True, help="docs-website checkout to update")
    sync_parser.set_defaults(func=sync_source)

    devnotes_parser = subparsers.add_parser("patch-devnotes")
    devnotes_parser.add_argument("--source-root", required=True, help="Repository checkout with latest Dev Notes")
    devnotes_parser.add_argument("--published-root", required=True, help="docs-website checkout to patch")
    devnotes_parser.set_defaults(func=patch_devnotes)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except PublishedBranchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
