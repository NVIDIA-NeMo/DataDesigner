# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[4] / "fern" / "scripts" / "fern-published-branch.py"


def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fern_published_branch", SCRIPT_PATH)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def patch_args(source_root: Path, published_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=source_root,
        published_root=published_root,
        metadata_source_repository=None,
        metadata_source_ref=None,
        metadata_source_sha=None,
        metadata_release_tag=None,
        metadata_published_branch=None,
    )


def test_patch_devnotes_syncs_root_config_and_preserves_published_versions(tmp_path: Path) -> None:
    module = load_script_module()
    source_root = tmp_path / "source"
    published_root = tmp_path / "published"

    write_text(
        source_root / "fern" / "docs.yml",
        """instances:
- url: datadesigner.docs.buildwithfern.com/nemo/datadesigner
title: Source Fern Docs
global-theme: nvidia
navbar-links:
- type: github
  value: https://github.com/NVIDIA-NeMo/DataDesigner
versions:
- display-name: "Latest"
  path: versions/latest.yml
  slug: latest
redirects:
  - source: "/nemo/datadesigner/getting-started"
    destination: "/nemo/datadesigner/getting-started/welcome"
""",
    )
    write_text(source_root / "fern" / "fern.config.json", '{"organization": "nvidia", "version": "5.41.1"}\n')
    write_text(source_root / "fern" / "assets" / "current-devnote-asset.png", "new asset")
    write_text(source_root / "fern" / "components" / "Figure.tsx", "export const Figure = () => null;\n")
    write_text(
        source_root / "fern" / "versions" / "latest.yml",
        """navigation:
  - section: Dev Notes
    contents:
      - page: New Note
        path: ./latest/pages/devnotes/posts/new-note.mdx
  - section: Concepts
    contents: []
""",
    )
    write_text(source_root / "fern" / "versions" / "latest" / "pages" / "devnotes" / "posts" / "new-note.mdx", "# New")

    write_text(
        published_root / "fern" / "docs.yml",
        """instances:
- url: datadesigner.docs.buildwithfern.com/nemo/datadesigner
title: Published Fern Docs
footer: ./components/OldFooter.tsx
layout:
  searchbar-placement: header
versions:
- display-name: latest
  path: versions/latest.yml
  slug: latest
- display-name: "v0.6.0"
  path: versions/v0.6.0.yml
  slug: v0.6.0
redirects:
  - source: "/nemo/datadesigner/getting-started"
    destination: "/nemo/datadesigner/getting-started/welcome"
""",
    )
    write_text(published_root / "fern" / "fern.config.json", '{"organization": "nvidia", "version": "4.106.0"}\n')
    write_text(published_root / "fern" / "assets" / "published-only-asset.png", "old asset")
    write_text(
        published_root / "fern" / "versions" / "latest.yml",
        """navigation:
  - section: Dev Notes
    contents:
      - page: Old Note
        path: ./latest/pages/devnotes/posts/old-note.mdx
  - section: Concepts
    contents: []
""",
    )

    assert module.patch_devnotes(patch_args(source_root, published_root)) == 0

    published_docs = (published_root / "fern" / "docs.yml").read_text()
    assert "title: Source Fern Docs" in published_docs
    assert "global-theme: nvidia" in published_docs
    assert "navbar-links:" in published_docs
    assert "title: Published Fern Docs" not in published_docs
    assert "footer: ./components/OldFooter.tsx" not in published_docs
    assert "searchbar-placement: header" not in published_docs
    assert '- display-name: "Latest"' in published_docs
    assert 'display-name: "v0.6.0"' in published_docs
    assert (published_root / "fern" / "fern.config.json").read_text() == (
        '{"organization": "nvidia", "version": "5.41.1"}\n'
    )
    assert (published_root / "fern" / "assets" / "current-devnote-asset.png").read_text() == "new asset"
    assert (published_root / "fern" / "components" / "Figure.tsx").read_text() == (
        "export const Figure = () => null;\n"
    )
    assert not (published_root / "fern" / "assets" / "published-only-asset.png").exists()
    assert (published_root / "fern" / "versions" / "latest" / "pages" / "devnotes" / "posts" / "new-note.mdx").exists()
