#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve Fern docs with a local approximation of the NVIDIA global theme."""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

LOCAL_THEME_CONFIG = {
    "layout": {
        "searchbar-placement": "header",
        "page-width": "1376px",
        "sidebar-width": "248px",
        "content-width": "812px",
        "tabs-placement": "header",
        "hide-feedback": True,
    },
    "colors": {
        "accent-primary": {
            "dark": "#76B900",
            "light": "#004B31",
        },
        "background": {
            "dark": "#000000",
            "light": "#FFFFFF",
        },
    },
    "theme": {
        "page-actions": "toolbar",
        "footer-nav": "minimal",
    },
}

COMPONENT_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
DOCS_CONFIG_PATH = Path("docs.yml")
POLL_INTERVAL_SECONDS = 0.25

FileState = tuple[int, int, int]
SourceState = dict[Path, FileState]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="fern", help="Path to the Fern docs root")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to run")
    return args


def write_local_docs_config(root: Path, preview_root: Path) -> None:
    config = yaml.safe_load((root / "docs.yml").read_text(encoding="utf-8"))
    config.pop("global-theme", None)

    for key, value in LOCAL_THEME_CONFIG.items():
        config.setdefault(key, value)

    logo = dict(config.get("logo") or {})
    logo.setdefault("height", 20)
    config["logo"] = logo

    css = config.get("css")
    local_css = "./styles/local-preview.css"
    if css is None:
        config["css"] = [local_css]
    elif isinstance(css, list) and local_css not in css:
        config["css"] = [*css, local_css]
    elif isinstance(css, str) and css != local_css:
        config["css"] = [css, local_css]

    (preview_root / "docs.yml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def snapshot_source(root: Path) -> SourceState:
    state = {}
    for source in root.rglob("*"):
        try:
            if not source.is_file():
                continue
            stat = source.stat()
        except FileNotFoundError:
            continue
        state[source.relative_to(root)] = (stat.st_mtime_ns, stat.st_ino, stat.st_size)
    return state


def component_aliases(state: SourceState) -> dict[Path, Path]:
    aliases = {}
    for source in state:
        if source.parts[:1] != ("components",) or source.suffix not in COMPONENT_EXTENSIONS:
            continue
        alias = source.with_suffix("")
        if alias not in state:
            aliases[alias] = source
    return aliases


def copy_preview_file(root: Path, preview_root: Path, source: Path, target: Path | None = None) -> None:
    target = preview_root / (target or source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / source, target)


def remove_preview_file(preview_root: Path, relative_path: Path) -> None:
    target = preview_root / relative_path
    target.unlink(missing_ok=True)
    parent = target.parent
    while parent != preview_root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def build_preview_root(root: Path, preview_root: Path) -> SourceState:
    state = snapshot_source(root)
    for relative_path in sorted(state):
        if relative_path == DOCS_CONFIG_PATH:
            write_local_docs_config(root, preview_root)
        else:
            copy_preview_file(root, preview_root, relative_path)
    for alias, source in component_aliases(state).items():
        copy_preview_file(root, preview_root, source, alias)
    return state


def sync_preview_root(root: Path, preview_root: Path, previous_state: SourceState) -> SourceState:
    current_state = snapshot_source(root)
    previous_aliases = component_aliases(previous_state)
    current_aliases = component_aliases(current_state)
    removed_paths = previous_state.keys() - current_state.keys()
    removed_aliases = previous_aliases.keys() - current_aliases.keys()
    for relative_path in sorted(removed_paths | removed_aliases):
        remove_preview_file(preview_root, relative_path)

    changed_paths = {
        relative_path
        for relative_path, file_state in current_state.items()
        if previous_state.get(relative_path) != file_state
    }
    failed_paths = set()
    for relative_path in sorted(changed_paths):
        try:
            if relative_path == DOCS_CONFIG_PATH:
                write_local_docs_config(root, preview_root)
            else:
                copy_preview_file(root, preview_root, relative_path)
        except FileNotFoundError:
            failed_paths.add(relative_path)

    changed_aliases = {
        alias for alias, source in current_aliases.items() if alias not in previous_aliases or source in changed_paths
    }
    for alias in sorted(changed_aliases):
        source = current_aliases[alias]
        try:
            copy_preview_file(root, preview_root, source, alias)
        except FileNotFoundError:
            failed_paths.add(source)

    for relative_path in failed_paths:
        current_state.pop(relative_path, None)
    return current_state


def run_command(command: list[str], cwd: Path, root: Path, source_state: SourceState) -> int:
    process = subprocess.Popen(command, cwd=cwd)
    try:
        while True:
            try:
                return process.wait(timeout=POLL_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                source_state = sync_preview_root(root, cwd, source_state)
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        return process.wait()
    except Exception:
        process.send_signal(signal.SIGINT)
        process.wait()
        raise


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    with tempfile.TemporaryDirectory(prefix="fern-local-preview-") as temp_dir:
        preview_root = Path(temp_dir) / "fern"
        preview_root.mkdir()
        source_state = build_preview_root(root, preview_root)
        print(f"Using local Fern preview config at {preview_root / 'docs.yml'}", file=sys.stderr)
        return run_command(args.command, preview_root, root, source_state)


if __name__ == "__main__":
    raise SystemExit(main())
