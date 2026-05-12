#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare or verify Fern release version entries."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:[-.][0-9A-Za-z]+)*")


class ReleaseVersionError(RuntimeError):
    pass


def normalize_version(value: str) -> str:
    version = value.strip()
    if version.startswith("refs/tags/"):
        version = version.removeprefix("refs/tags/")
    version = version.removeprefix("v")
    if not VERSION_RE.fullmatch(version):
        raise ReleaseVersionError(f"Invalid version '{value}'. Expected X.Y.Z or vX.Y.Z, with optional suffix.")
    return version


def version_slug(version: str) -> str:
    return f"v{normalize_version(version)}"


def find_top_level_block(lines: list[str], name: str) -> tuple[int, int]:
    start = next((i for i, line in enumerate(lines) if line == f"{name}:\n"), -1)
    if start == -1:
        raise ReleaseVersionError(f"Missing top-level '{name}:' block")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z0-9_-]+:", lines[i]):
            end = i
            break
    return start, end


def read_docs_lines(root: Path) -> list[str]:
    docs_yml = root / "docs.yml"
    if not docs_yml.exists():
        raise ReleaseVersionError(f"Missing {docs_yml}")
    return docs_yml.read_text().splitlines(keepends=True)


def versions_block_text(root: Path) -> str:
    lines = read_docs_lines(root)
    start, end = find_top_level_block(lines, "versions")
    return "".join(lines[start:end])


def has_version_entry(root: Path, slug: str) -> bool:
    block = versions_block_text(root)
    return re.search(rf"^\s+slug:\s+{re.escape(slug)}\s*$", block, re.MULTILINE) is not None


def update_docs_yml(root: Path, slug: str) -> None:
    docs_yml = root / "docs.yml"
    lines = read_docs_lines(root)
    start, end = find_top_level_block(lines, "versions")

    latest_index = next(
        (i for i in range(start + 1, end) if lines[i].startswith("- display-name:") and "Latest" in lines[i]),
        -1,
    )
    if latest_index == -1:
        raise ReleaseVersionError("Missing latest version entry in docs.yml")
    lines[latest_index] = f'- display-name: "Latest · {slug}"\n'

    if not has_version_entry(root, slug):
        insert_index = end
        for i in range(latest_index + 1, end):
            if lines[i].startswith("- display-name:"):
                insert_index = i
                break
        lines[insert_index:insert_index] = [
            f'- display-name: "{slug}"\n',
            f"  path: versions/{slug}.yml\n",
            f"  slug: {slug}\n",
        ]

    docs_yml.write_text("".join(lines))


def strip_leading_comment_block(content: str) -> str:
    lines = content.splitlines(keepends=True)
    index = 0
    while index < len(lines) and (lines[index].startswith("#") or not lines[index].strip()):
        index += 1
    return "".join(lines[index:])


def write_release_nav(root: Path, slug: str, force: bool) -> bool:
    versions_dir = root / "versions"
    source = versions_dir / "latest.yml"
    target = versions_dir / f"{slug}.yml"
    if not source.exists():
        raise ReleaseVersionError(f"Missing {source}")
    if target.exists() and not force:
        raise ReleaseVersionError(f"{target} already exists. Pass --force to replace it.")

    content = source.read_text()
    copied_pages = False
    if "./latest/pages/" in content:
        source_pages = versions_dir / "latest" / "pages"
        target_pages = versions_dir / slug / "pages"
        if not source_pages.exists():
            raise ReleaseVersionError(f"{source_pages} is referenced by latest.yml but does not exist")
        if target_pages.exists() and not force:
            raise ReleaseVersionError(f"{target_pages} already exists. Pass --force to replace it.")
        if target_pages.exists():
            shutil.rmtree(target_pages)
        shutil.copytree(source_pages, target_pages)
        copied_pages = True
        content = content.replace("./latest/pages/", f"./{slug}/pages/")

    release_comment = f"# Frozen {slug} release nav. Reuses shared pages until content needs to diverge.\n"
    target.write_text(release_comment + strip_leading_comment_block(content))
    return copied_pages


def check_release(root: Path, slug: str) -> list[str]:
    errors: list[str] = []
    block = versions_block_text(root)
    nav = root / "versions" / f"{slug}.yml"

    expected = {
        "latest display name": rf'^- display-name:\s+["\']Latest\b.*{re.escape(slug)}["\']\s*$',
        "version display name": rf'^- display-name:\s+["\']{re.escape(slug)}["\']\s*$',
        "version path": rf"^\s+path:\s+versions/{re.escape(slug)}\.yml\s*$",
        "version slug": rf"^\s+slug:\s+{re.escape(slug)}\s*$",
    }
    for label, pattern in expected.items():
        if not re.search(pattern, block, re.MULTILINE):
            errors.append(f"Missing {label} for {slug} in docs.yml")

    if not nav.exists():
        errors.append(f"Missing {nav}")
    elif "navigation:" not in nav.read_text():
        errors.append(f"{nav} does not look like a Fern version nav file")

    return errors


def prepare(args: argparse.Namespace) -> int:
    root = Path(args.root)
    slug = version_slug(args.version)
    copied_pages = write_release_nav(root, slug, args.force)
    update_docs_yml(root, slug)
    print(f"Prepared Fern release {slug}")
    if copied_pages:
        print(f"Copied latest-only pages into {root / 'versions' / slug / 'pages'}")
    else:
        print("No latest-only pages needed copying")
    print("Review reused page paths before publishing the release.")
    return 0


def check(args: argparse.Namespace) -> int:
    root = Path(args.root)
    slug = version_slug(args.version)
    errors = check_release(root, slug)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Fern release version is prepared for {slug}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="fern", help="Fern docs root")
    subparsers = parser.add_subparsers(required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare Fern files for a release")
    prepare_parser.add_argument("--version", required=True, help="Release version, e.g. 0.5.10")
    prepare_parser.add_argument("--force", action="store_true", help="Overwrite existing release files")
    prepare_parser.set_defaults(func=prepare)

    check_parser = subparsers.add_parser("check", help="Check Fern files include a release")
    check_parser.add_argument("--version", required=True, help="Release version or tag, e.g. v0.5.10")
    check_parser.set_defaults(func=check)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ReleaseVersionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
