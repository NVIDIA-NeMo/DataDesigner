# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT_PATH = Path(__file__).parents[1] / "check-internal-links.py"
SPEC = importlib.util.spec_from_file_location("check_internal_links", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_internal_links = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_internal_links
SPEC.loader.exec_module(check_internal_links)


@pytest.fixture
def docs_root(tmp_path: Path) -> Path:
    root = tmp_path / "fern"
    pages = root / "versions/latest/pages"
    pages.mkdir(parents=True)
    (root / "docs.yml").write_text(
        yaml.safe_dump(
            {
                "instances": [{"custom-domain": "docs.example.com/nemo/datadesigner"}],
                "versions": [{"display-name": "Latest", "path": "versions/latest.yml", "slug": "latest"}],
                "redirects": [
                    {
                        "source": "/nemo/datadesigner/old/:path*",
                        "destination": "/nemo/datadesigner/concepts/columns",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "versions/latest.yml").write_text(
        yaml.safe_dump(
            {
                "navigation": [
                    {
                        "section": "Concepts",
                        "contents": [
                            {"page": "Columns", "path": "./latest/pages/columns.mdx"},
                            {
                                "section": "Tool Use & MCP",
                                "contents": [{"page": "Safety & Limits", "path": "./latest/pages/safety.mdx"}],
                            },
                        ],
                    },
                    {
                        "section": "Tutorials",
                        "contents": [{"page": "The Basics", "path": "./latest/pages/notebooks/the-basics.mdx"}],
                    },
                    {
                        "section": "Dev Notes",
                        "contents": [
                            {
                                "section": "Older Posts",
                                "skip-slug": True,
                                "contents": [{"page": "Design Principles", "path": "./latest/pages/design.mdx"}],
                            }
                        ],
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (pages / "columns.mdx").write_text(
        "# Columns\n\n## LLM columns\n\n[Safety](/concepts/tool-use-mcp/safety-limits#budgets)\n",
        encoding="utf-8",
    )
    (pages / "safety.mdx").write_text("# Safety & Limits\n\n## Budgets\n", encoding="utf-8")
    (pages / "design.mdx").write_text("# Design Principles\n", encoding="utf-8")
    (pages / "notebooks").mkdir()
    (pages / "notebooks/the-basics.mdx").write_text("# The Basics\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Architecture & Performance", "architecture-performance"),
        ("Tool Use & MCP", "tool-use-mcp"),
        ("Text-to-SQL for Nemotron Super", "text-to-sql-for-nemotron-super"),
        ("FileSystemSeedReader Plugins", "file-system-seed-reader-plugins"),
        ("What's New?", "whats-new"),
    ],
)
def test_slugify_matches_fern_routes(title: str, expected: str) -> None:
    assert check_internal_links.slugify(title) == expected


def test_anchor_slugify_preserves_camel_case_boundaries() -> None:
    assert check_internal_links.anchor_slugify("LocalStdioMCPProvider (Subprocess)") == (
        "localstdiomcpprovider-subprocess"
    )


def test_build_index_uses_navigation_titles_and_skip_slug(docs_root: Path) -> None:
    index = check_internal_links.build_index(docs_root)

    assert "/concepts/columns" in index.routes
    assert "/concepts/tool-use-mcp/safety-limits" in index.routes
    assert "/dev-notes/design-principles" in index.routes
    assert "/dev-notes/older-posts/design-principles" not in index.routes
    assert "/latest/concepts/columns" in index.routes
    assert index.routes["/"].route == "/concepts/columns"


def test_validate_links_accepts_routes_fragments_relative_links_and_redirects(docs_root: Path) -> None:
    columns = docs_root / "versions/latest/pages/columns.mdx"
    columns.write_text(
        "# Columns\n\n"
        "[Current fragment](#llm-columns)\n\n"
        "## LLM columns\n\n"
        "[Nested route](/concepts/tool-use-mcp/safety-limits#budgets)\n\n"
        "[Version route](/latest/concepts/columns)\n\n"
        "[Base-prefixed route](/nemo/datadesigner/concepts/columns)\n\n"
        "[Configured redirect](/old/legacy-page)\n",
        encoding="utf-8",
    )
    index = check_internal_links.build_index(docs_root)

    assert check_internal_links.validate_links(index) == []


def test_validate_links_reports_missing_route_with_suggestion(docs_root: Path) -> None:
    columns = docs_root / "versions/latest/pages/columns.mdx"
    columns.write_text("# Columns\n\n[Broken](/concepts/tool-use-and-mcp/safety-and-limits)\n", encoding="utf-8")
    index = check_internal_links.build_index(docs_root)

    errors = check_internal_links.validate_links(index)

    assert len(errors) == 1
    assert errors[0].link.line == 3
    assert "internal target does not exist" in errors[0].message
    assert "/concepts/tool-use-mcp/safety-limits" in errors[0].message


def test_validate_links_reports_missing_fragment(docs_root: Path) -> None:
    columns = docs_root / "versions/latest/pages/columns.mdx"
    columns.write_text("# Columns\n\n[Broken](#missing)\n", encoding="utf-8")
    index = check_internal_links.build_index(docs_root)

    errors = check_internal_links.validate_links(index)

    assert len(errors) == 1
    assert errors[0].message == "fragment does not exist on /concepts/columns: #missing"


def test_validate_links_ignores_code_blocks_and_external_urls(docs_root: Path) -> None:
    columns = docs_root / "versions/latest/pages/columns.mdx"
    columns.write_text(
        "# Columns\n\n[External](https://example.com/missing)\n\n```markdown\n[Example](/does-not-exist)\n```\n",
        encoding="utf-8",
    )
    index = check_internal_links.build_index(docs_root)

    assert check_internal_links.validate_links(index) == []


def test_notebook_sources_add_generated_heading_anchors(docs_root: Path, tmp_path: Path) -> None:
    notebook_source = tmp_path / "docs/notebook_source/the-basics.py"
    notebook_source.parent.mkdir(parents=True)
    notebook_source.write_text("# %% [markdown]\n# ## Generated heading\n", encoding="utf-8")
    columns = docs_root / "versions/latest/pages/columns.mdx"
    columns.write_text("# Columns\n\n[Notebook heading](/tutorials/the-basics#generated-heading)\n", encoding="utf-8")
    index = check_internal_links.build_index(docs_root)

    extra_sources = check_internal_links.notebook_sources(index, tmp_path)

    assert check_internal_links.validate_links(index, extra_sources) == []


def test_notebook_markdown_extracts_only_markdown_cells() -> None:
    source = "# %% [markdown]\n# ## Included heading\n#\n# Text\n# %%\n# ## Python comment\nvalue = 1\n"

    assert check_internal_links.notebook_markdown(source) == "## Included heading\n\nText\n"
