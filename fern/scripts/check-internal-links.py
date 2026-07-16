# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import difflib
import html
import posixpath
import re
import sys
import unicodedata
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MARKDOWN_LINK_PATTERN = re.compile(
    r"(?<!!)\[[^\]]*\]\(\s*(?:<(?P<angle>[^>\n]+)>|(?P<plain>[^)\s]+))"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)",
    re.MULTILINE,
)
HREF_PATTERN = re.compile(r"\bhref\s*=\s*(?:\{\s*)?([\"'])(?P<target>.+?)\1\s*\}?", re.MULTILINE)
EXPLICIT_ID_PATTERN = re.compile(r"\bid\s*=\s*(?:\{\s*)?([\"'])(?P<identifier>.+?)\1\s*\}?")
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)
FENCE_PATTERN = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})")
INLINE_CODE_PATTERN = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
MARKDOWN_DECORATION_PATTERN = re.compile(r"[*_~]|<[^>]+>|!?(?:\[([^\]]*)\]\([^)]+\))")
EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}
DOCS_HOSTS = {"docs.nvidia.com", "datadesigner.docs.buildwithfern.com"}


@dataclass
class Page:
    source: Path
    route: str
    version_slug: str
    anchors: set[str]


@dataclass(frozen=True)
class Link:
    source: Path
    line: int
    source_route: str
    target: str


@dataclass(frozen=True)
class LinkError:
    link: Link
    message: str


@dataclass
class DocsIndex:
    root: Path
    base_path: str
    pages: list[Page] = field(default_factory=list)
    routes: dict[str, Page] = field(default_factory=dict)
    redirects: list[str] = field(default_factory=list)


def slugify(value: str) -> str:
    """Approximate Fern's title-to-route conversion."""
    normalized = unicodedata.normalize("NFKD", html.unescape(value)).encode("ascii", "ignore").decode()
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"['’]", "", normalized.casefold())
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def anchor_slugify(value: str) -> str:
    """Convert a rendered heading to the anchor format used by Fern."""
    normalized = unicodedata.normalize("NFKD", html.unescape(value)).encode("ascii", "ignore").decode()
    normalized = re.sub(r"['’]", "", normalized.casefold())
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def strip_code(text: str) -> str:
    """Mask fenced and inline code while preserving offsets and line numbers."""
    masked_lines: list[str] = []
    active_fence: str | None = None
    for line in text.splitlines(keepends=True):
        fence_match = FENCE_PATTERN.match(line)
        if fence_match:
            marker = fence_match.group("fence")
            if active_fence is None:
                active_fence = marker[0]
            elif marker[0] == active_fence:
                active_fence = None
            masked_lines.append("".join("\n" if character == "\n" else " " for character in line))
        elif active_fence is not None:
            masked_lines.append("".join("\n" if character == "\n" else " " for character in line))
        else:
            masked_lines.append(line)
    return INLINE_CODE_PATTERN.sub(lambda match: " " * len(match.group(0)), "".join(masked_lines))


def heading_anchor(title: str) -> str:
    title = MARKDOWN_DECORATION_PATTERN.sub(lambda match: match.group(1) or "", title)
    return anchor_slugify(title)


def extract_anchors(text: str) -> set[str]:
    masked = strip_code(text)
    anchors = {match.group("identifier") for match in EXPLICIT_ID_PATTERN.finditer(masked)}
    counts: dict[str, int] = defaultdict(int)
    for match in HEADING_PATTERN.finditer(masked):
        anchor = heading_anchor(match.group("title"))
        if not anchor:
            continue
        duplicate = counts[anchor]
        anchors.add(anchor if duplicate == 0 else f"{anchor}-{duplicate}")
        counts[anchor] += 1
    return anchors


def extract_links(source: Path, source_route: str) -> list[Link]:
    text = source.read_text(encoding="utf-8")
    masked = strip_code(text)
    matches: list[tuple[int, str]] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(masked):
        matches.append((match.start(), match.group("angle") or match.group("plain")))
    for match in HREF_PATTERN.finditer(masked):
        matches.append((match.start(), match.group("target")))
    return [
        Link(
            source=source,
            line=text.count("\n", 0, offset) + 1,
            source_route=source_route,
            target=html.unescape(target),
        )
        for offset, target in sorted(set(matches))
    ]


def version_routes(version_config: dict[str, Any], version_slug: str, config_path: Path) -> list[Page]:
    pages: list[Page] = []

    def walk(items: list[dict[str, Any]], parents: tuple[str, ...]) -> None:
        for item in items:
            if "section" in item:
                section_parents = parents
                if not item.get("skip-slug", False):
                    section_parents += (item.get("slug") or slugify(str(item["section"])),)
                walk(item.get("contents", []), section_parents)
                continue
            if "page" not in item or "path" not in item:
                continue
            page_slug = item.get("slug") or slugify(str(item["page"]))
            relative_route = "/" + "/".join((*parents, page_slug))
            route = relative_route if version_slug == "latest" else f"/{version_slug}{relative_route}"
            source = (config_path.parent / str(item["path"])).resolve()
            anchors = extract_anchors(source.read_text(encoding="utf-8")) if source.is_file() else set()
            pages.append(Page(source=source, route=route, version_slug=version_slug, anchors=anchors))

    walk(version_config.get("navigation", []), ())
    return pages


def docs_base_path(docs_config: dict[str, Any]) -> str:
    instances = docs_config.get("instances", [])
    if not instances:
        return ""
    instance_url = str(instances[0].get("custom-domain") or instances[0].get("url") or "")
    parsed = urllib.parse.urlsplit(instance_url if "://" in instance_url else f"https://{instance_url}")
    return parsed.path.rstrip("/")


def build_index(root: Path, version_slugs: set[str] | None = None) -> DocsIndex:
    root = root.resolve()
    docs_config = yaml.safe_load((root / "docs.yml").read_text(encoding="utf-8"))
    index = DocsIndex(root=root, base_path=docs_base_path(docs_config))
    configured_versions = docs_config.get("versions", [])
    configured_slugs = {str(version["slug"]) for version in configured_versions}
    if version_slugs is not None:
        unknown_slugs = version_slugs - configured_slugs
        if unknown_slugs:
            raise ValueError(f"unknown Fern version slug(s): {', '.join(sorted(unknown_slugs))}")
    for version in configured_versions:
        if version_slugs is not None and str(version["slug"]) not in version_slugs:
            continue
        version_config_path = (root / str(version["path"])).resolve()
        version_config = yaml.safe_load(version_config_path.read_text(encoding="utf-8"))
        pages = version_routes(version_config, str(version["slug"]), version_config_path)
        index.pages.extend(pages)
        for page in pages:
            index.routes[page.route] = page
            if page.version_slug == "latest":
                index.routes[f"/latest{page.route}"] = page
    latest_pages = [page for page in index.pages if page.version_slug == "latest"]
    if latest_pages:
        index.routes["/"] = latest_pages[0]
        index.routes["/latest"] = latest_pages[0]
    index.redirects = [
        normalize_configured_path(str(item["source"]), index.base_path) for item in docs_config.get("redirects", [])
    ]
    return index


def normalize_configured_path(path: str, base_path: str) -> str:
    parsed = urllib.parse.urlsplit(path)
    if parsed.scheme or parsed.netloc:
        return path
    normalized = parsed.path or "/"
    if base_path and (normalized == base_path or normalized.startswith(f"{base_path}/")):
        normalized = normalized[len(base_path) :] or "/"
    return normalized.rstrip("/") or "/"


def redirect_matches(path: str, redirect: str) -> bool:
    expression = re.escape(redirect)
    expression = re.sub(r":([A-Za-z][A-Za-z0-9_]*)\\\*", r".*", expression)
    expression = re.sub(r":([A-Za-z][A-Za-z0-9_]*)", r"[^/]+", expression)
    return re.fullmatch(expression, path) is not None


def normalize_target(link: Link, index: DocsIndex) -> tuple[str, str] | None:
    parsed = urllib.parse.urlsplit(link.target)
    if parsed.scheme in EXTERNAL_SCHEMES:
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in DOCS_HOSTS:
            return None
        path = parsed.path
    elif parsed.scheme or parsed.netloc:
        return None
    else:
        path = parsed.path
    if not path:
        normalized_path = link.source_route
    elif path.startswith("/"):
        normalized_path = normalize_configured_path(path, index.base_path)
    else:
        normalized_path = posixpath.normpath(posixpath.join(posixpath.dirname(link.source_route), path))
        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"
    return normalized_path.rstrip("/") or "/", urllib.parse.unquote(parsed.fragment)


def asset_exists(path: str, index: DocsIndex) -> bool:
    relative_path = path.lstrip("/")
    if not relative_path or relative_path.startswith("."):
        return False
    return (index.root / relative_path).is_file()


def closest_route(path: str, routes: dict[str, Page]) -> str | None:
    matches = difflib.get_close_matches(path, routes, n=1, cutoff=0.55)
    return matches[0] if matches else None


def validate_links(index: DocsIndex, extra_sources: list[tuple[Path, str]] | None = None) -> list[LinkError]:
    sources = [(page.source, page.route) for page in index.pages]
    if extra_sources:
        sources.extend(extra_sources)
    errors: list[LinkError] = []
    seen: set[tuple[Path, int, str]] = set()
    for source, source_route in sources:
        if not source.is_file():
            errors.append(LinkError(Link(source, 1, source_route, ""), "navigation points to a missing source file"))
            continue
        for link in extract_links(source, source_route):
            identity = (link.source, link.line, link.target)
            if identity in seen:
                continue
            seen.add(identity)
            normalized = normalize_target(link, index)
            if normalized is None:
                continue
            target_path, fragment = normalized
            page = index.routes.get(target_path)
            if page is None:
                if asset_exists(target_path, index) or any(
                    redirect_matches(target_path, redirect) for redirect in index.redirects
                ):
                    continue
                suggestion = closest_route(target_path, index.routes)
                message = f"internal target does not exist: {target_path}"
                if suggestion:
                    message += f" (did you mean {suggestion}?)"
                errors.append(LinkError(link, message))
                continue
            if fragment and fragment not in page.anchors:
                errors.append(LinkError(link, f"fragment does not exist on {target_path}: #{fragment}"))
    return errors


def notebook_markdown(text: str) -> str:
    markdown_lines: list[str] = []
    in_markdown_cell = False
    for line in text.splitlines(keepends=True):
        if line.startswith("# %%"):
            in_markdown_cell = "[markdown]" in line
            continue
        if not in_markdown_cell:
            continue
        if line.startswith("# "):
            markdown_lines.append(line[2:])
        elif line.startswith("#"):
            markdown_lines.append(line[1:])
    return "".join(markdown_lines)


def notebook_sources(index: DocsIndex, repository_root: Path) -> list[tuple[Path, str]]:
    wrapper_pages = {page.source.stem: page for page in index.pages if "/notebooks/" in page.source.as_posix()}
    sources: list[tuple[Path, str]] = []
    for source in sorted((repository_root / "docs/notebook_source").glob("*.py")):
        wrapper_stem = "README" if source.stem in {"README", "_README"} else source.stem
        page = wrapper_pages.get(wrapper_stem)
        if page:
            text = source.read_text(encoding="utf-8")
            page.anchors.update(extract_anchors(notebook_markdown(text)))
            sources.append((source, page.route))
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate internal links against Fern navigation-derived routes.")
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1], help="Fern documentation root")
    parser.add_argument(
        "--version",
        dest="versions",
        action="append",
        help="Fern version slug to validate (repeatable; validates all configured versions when omitted)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index = build_index(args.root, set(args.versions) if args.versions else None)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    repository_root = index.root.parent
    errors = validate_links(index, notebook_sources(index, repository_root))
    if not errors:
        print(f"Checked internal links across {len(index.pages)} Fern pages.")
        return 0
    print(f"Found {len(errors)} broken internal link(s):", file=sys.stderr)
    for error in errors:
        relative_source = error.link.source.relative_to(repository_root)
        print(f"  {relative_source}:{error.link.line}: {error.message} [{error.link.target}]", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
