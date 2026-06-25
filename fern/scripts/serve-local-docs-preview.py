#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve Fern docs with a local approximation of the NVIDIA global theme."""

from __future__ import annotations

import argparse
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


def build_preview_root(root: Path, preview_root: Path) -> None:
    for child in root.iterdir():
        if child.name == "docs.yml":
            continue
        target = preview_root / child.name
        target.symlink_to(child, target_is_directory=child.is_dir())
    write_local_docs_config(root, preview_root)


def run_command(command: list[str], cwd: Path) -> int:
    process = subprocess.Popen(command, cwd=cwd)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        return process.wait()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    with tempfile.TemporaryDirectory(prefix="fern-local-preview-") as temp_dir:
        preview_root = Path(temp_dir) / "fern"
        preview_root.mkdir()
        build_preview_root(root, preview_root)
        print(f"Using local Fern preview config at {preview_root / 'docs.yml'}", file=sys.stderr)
        return run_command(args.command, preview_root)


if __name__ == "__main__":
    raise SystemExit(main())
