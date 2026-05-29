#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync Fern authoring content into the CI-managed publish branch."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

DEVNOTES_SECTION_RE = re.compile(r"^  - section:\s+Dev Notes\s*$")
RETIRED_REFERENCE_SECTION_RE = re.compile(r"^  - section:\s+" + re.escape("Code " + "Reference") + r"\s*$")
RETIRED_REFERENCE_DIR = "code" + "_reference"
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
    "__pycache__",
    "dist",
    "site",
}
PUBLISH_METADATA_PATH = Path("fern/publish-metadata.json")
# Dev-note kit components carry their own CSS (injected via a <style> tag), so no
# stylesheet files are copied here — `css` is a theme-owned field and any local
# `css:` list is dropped under `global-theme: nvidia`. See fern/docs.yml.
FERN_DEVNOTE_SUPPORT_PATHS = [
    "fern/assets",
    "fern/components/Authors.tsx",
    "fern/components/BlogCard.tsx",
    "fern/components/MetricsTable.tsx",
    "fern/components/TrajectoryViewer.tsx",
    "fern/components/devnotes",
]
RETIRED_REFERENCE_CLEAN_PAGE_PATHS = [
    "concepts/columns.mdx",
    "concepts/custom_columns.mdx",
    "concepts/models/model-configs.mdx",
    "concepts/person_sampling.mdx",
    "concepts/security.mdx",
    "concepts/tool_use_and_mcp.mdx",
    "concepts/validators.mdx",
    "plugins/example.mdx",
    "plugins/overview.mdx",
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


def normalize_latest_display_name(block: list[str] | None) -> list[str] | None:
    if block is None:
        return None

    normalized = list(block)
    display_name_index = -1
    for index, line in enumerate(block):
        if line.startswith("- display-name:"):
            display_name_index = index
            continue
        match = VERSION_SLUG_RE.match(line)
        if display_name_index != -1 and match and match.group(1) == "latest":
            normalized[display_name_index] = '- display-name: "Latest"\n'
            break
    return normalized


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


def write_publish_metadata(published_root: Path, args: argparse.Namespace, action: str) -> None:
    provided = [
        args.metadata_source_repository,
        args.metadata_source_ref,
        args.metadata_source_sha,
        args.metadata_release_tag,
        args.metadata_published_branch,
    ]
    if not any(provided):
        return

    missing = [
        name
        for name, value in (
            ("metadata source repository", args.metadata_source_repository),
            ("metadata source ref", args.metadata_source_ref),
            ("metadata source sha", args.metadata_source_sha),
        )
        if not value
    ]
    if missing:
        raise PublishedBranchError(f"Incomplete publish metadata; missing {', '.join(missing)}")

    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "fern-docs-website",
        "action": action,
        "source": {
            "repository": args.metadata_source_repository,
            "ref": args.metadata_source_ref,
            "sha": args.metadata_source_sha,
        },
    }
    if args.metadata_release_tag:
        metadata["release_tag"] = args.metadata_release_tag
    if args.metadata_published_branch:
        metadata["published_branch"] = args.metadata_published_branch

    target = published_root / PUBLISH_METADATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2) + "\n")


def ignore_source(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_NAMES}


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


def extract_navigation_section(path: Path, section_re: re.Pattern[str]) -> list[str]:
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if section_re.match(line)), -1)
    if start == -1:
        raise PublishedBranchError(f"Section not found in {path}")
    end = start + 1
    while end < len(lines):
        if lines[end].startswith("  - ") and lines[end].strip():
            break
        end += 1
    return lines[start:end]


def replace_navigation_section(path: Path, section_re: re.Pattern[str], block: list[str]) -> None:
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if section_re.match(line)), -1)
    if start == -1:
        raise PublishedBranchError(f"Section not found in {path}")
    end = start + 1
    while end < len(lines):
        if lines[end].startswith("  - ") and lines[end].strip():
            break
        end += 1
    lines[start:end] = block
    path.write_text("".join(lines))


def remove_navigation_section(path: Path, section_re: re.Pattern[str]) -> None:
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if section_re.match(line)), -1)
    if start == -1:
        return
    end = start + 1
    while end < len(lines):
        if lines[end].startswith("  - ") and lines[end].strip():
            break
        end += 1
    lines[start:end] = []
    path.write_text("".join(lines))


def remove_retired_reference_archive(source_root: Path, published_root: Path) -> None:
    versions_dir = published_root / "fern" / "versions"
    for nav in sorted(path for path in versions_dir.glob("*.yml") if path.name != "latest.yml"):
        remove_navigation_section(nav, RETIRED_REFERENCE_SECTION_RE)

    for path in sorted(versions_dir.glob(f"*/pages/{RETIRED_REFERENCE_DIR}")):
        if path.is_dir():
            shutil.rmtree(path)

    source_pages = source_root / "fern" / "versions" / "latest" / "pages"
    for pages_dir in sorted(versions_dir.glob("v*/pages")):
        for rel_path in RETIRED_REFERENCE_CLEAN_PAGE_PATHS:
            source_file = source_pages / rel_path
            target_file = pages_dir / rel_path
            if source_file.exists() and target_file.exists():
                copy_path(source_file, target_file)


def materialize_version_nav_pages(published_root: Path) -> None:
    versions_dir = published_root / "fern" / "versions"
    for nav in sorted(versions_dir.glob("v*.yml")):
        slug = nav.stem
        lines = nav.read_text().splitlines(keepends=True)
        changed = False
        if lines and lines[0].startswith(f"# Frozen {slug} release nav. Reuses shared pages"):
            lines[0] = f"# Frozen {slug} release nav. Pages are materialized under ./{slug}/pages/.\n"
            changed = True
        for index, line in enumerate(lines):
            match = NAV_PATH_RE.match(line)
            if not match:
                continue
            rel_path = Path(match.group(2))
            if len(rel_path.parts) < 3 or rel_path.parts[1] != "pages":
                continue

            target_rel = Path(slug, "pages", *rel_path.parts[2:])
            source_file = versions_dir / rel_path
            target_file = versions_dir / target_rel
            if not source_file.exists():
                raise PublishedBranchError(f"{nav} references missing page {source_file}")
            if source_file != target_file:
                copy_path(source_file, target_file)
            lines[index] = f"{match.group(1)}./{target_rel.as_posix()}{match.group(3)}\n"
            changed = True

        if changed:
            nav.write_text("".join(lines))


def sync_source(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root)
    published_root = Path(args.published_root)
    if not (source_root / "fern" / "docs.yml").exists():
        raise PublishedBranchError(f"Missing source Fern docs at {source_root / 'fern'}")

    preserved_versions_block = normalize_latest_display_name(versions_block(published_root / "fern" / "docs.yml"))
    with tempfile.TemporaryDirectory() as tmpdir:
        preserved_versions = Path(tmpdir) / "versions"
        if (published_root / "fern" / "versions").exists():
            shutil.copytree(published_root / "fern" / "versions", preserved_versions)

        clear_published_tree(published_root)
        shutil.copytree(source_root, published_root, dirs_exist_ok=True, ignore=ignore_source)
        merge_preserved_versions(
            source_root / "fern" / "versions", published_root / "fern" / "versions", preserved_versions
        )
        remove_retired_reference_archive(source_root, published_root)
        materialize_version_nav_pages(published_root)
        restore_versions_block(published_root / "fern" / "docs.yml", preserved_versions_block)
        validate_redirect_targets(published_root)
        write_publish_metadata(published_root, args, "release-snapshot")
    return 0


def extract_devnotes_block(path: Path) -> list[str]:
    return extract_navigation_section(path, DEVNOTES_SECTION_RE)


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
    replace_navigation_section(path, DEVNOTES_SECTION_RE, block)


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
    write_publish_metadata(published_root, args, "devnotes-patch")
    return 0


def add_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata-source-repository", help="Repository used to produce this published snapshot")
    parser.add_argument("--metadata-source-ref", help="Git ref used to produce this published snapshot")
    parser.add_argument("--metadata-source-sha", help="Git commit used to produce this published snapshot")
    parser.add_argument("--metadata-release-tag", help="Release tag represented by this published snapshot")
    parser.add_argument("--metadata-published-branch", help="Published branch updated by this snapshot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    sync_parser = subparsers.add_parser("sync-source")
    sync_parser.add_argument("--source-root", required=True, help="Repository checkout with authoring content")
    sync_parser.add_argument("--published-root", required=True, help="docs-website checkout to update")
    add_metadata_args(sync_parser)
    sync_parser.set_defaults(func=sync_source)

    devnotes_parser = subparsers.add_parser("patch-devnotes")
    devnotes_parser.add_argument("--source-root", required=True, help="Repository checkout with latest Dev Notes")
    devnotes_parser.add_argument("--published-root", required=True, help="docs-website checkout to patch")
    add_metadata_args(devnotes_parser)
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
