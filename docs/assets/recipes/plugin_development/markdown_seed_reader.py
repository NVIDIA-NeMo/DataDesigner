# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Markdown Section Seed Reader Recipe

Prototype a custom FileSystemSeedReader inline by overriding how one
DataDesigner instance handles DirectorySeedSource inputs. The reader keeps a
file-based manifest and fans each Markdown file out into one row per section.
This keeps the example in the same single-file format as the other recipes
while still showing the core `build_manifest(...)` and `hydrate_row(...)`
contract for a custom filesystem-backed seed reader.

Run:
    uv run markdown_seed_reader.py
"""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import data_designer.config as dd
from data_designer.config.seed import IndexRange
from data_designer.engine.resources.seed_reader import FileSystemSeedReader, SeedReaderFileSystemContext
from data_designer.interface import DataDesigner

_ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")


class MarkdownSectionDirectorySeedReader(FileSystemSeedReader[dd.DirectorySeedSource]):
    """Turn each Markdown file matched by DirectorySeedSource into section rows."""

    output_columns: ClassVar[list[str]] = [
        "relative_path",
        "file_name",
        "section_index",
        "section_header",
        "section_content",
    ]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, str]]:
        """Return one cheap manifest row per matched Markdown file."""

        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                "file_name": Path(relative_path).name,
            }
            for relative_path in matched_paths
        ]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        """Read one Markdown file and fan it out into one record per heading section."""

        relative_path = str(manifest_row["relative_path"])
        file_name = str(manifest_row["file_name"])
        with context.fs.open(relative_path, "r", encoding="utf-8") as handle:
            markdown_text = handle.read()

        sections = extract_markdown_sections(markdown_text=markdown_text, fallback_header=file_name)
        return [
            {
                "relative_path": relative_path,
                "file_name": file_name,
                "section_index": section_index,
                "section_header": section_header,
                "section_content": section_content,
            }
            for section_index, (section_header, section_content) in enumerate(sections)
        ]


def extract_markdown_sections(*, markdown_text: str, fallback_header: str) -> list[tuple[str, str]]:
    """Split Markdown into `(header, content)` pairs using ATX headings."""

    sections: list[tuple[str, str]] = []
    current_header = fallback_header
    current_lines: list[str] = []
    saw_heading = False

    for line in markdown_text.splitlines():
        heading_match = _ATX_HEADING_PATTERN.match(line)
        if heading_match is not None:
            if saw_heading or any(existing_line.strip() for existing_line in current_lines):
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = heading_match.group(2).strip()
            current_lines = []
            saw_heading = True
            continue
        current_lines.append(line)

    if saw_heading or markdown_text.strip():
        sections.append((current_header, "\n".join(current_lines).strip()))

    return [
        (section_header, section_content)
        for section_header, section_content in sections
        if section_header or section_content
    ]


def create_sample_markdown_files(seed_dir: Path) -> None:
    """Create a tiny Markdown corpus that keeps the recipe self-contained."""

    (seed_dir / "faq.md").write_text(
        "# FAQ\nAnswers to frequent questions.\n\n## Support\nContact support@example.com.",
        encoding="utf-8",
    )
    (seed_dir / "guide.md").write_text(
        "# Quickstart\nInstall Data Designer.\n\n## Usage\nRun the recipe with uv.",
        encoding="utf-8",
    )


def build_config(
    *,
    seed_path: Path,
    selection_strategy: IndexRange | None = None,
) -> dd.DataDesignerConfigBuilder:
    """Create the dataset config used by both preview runs in the recipe."""

    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(
        dd.DirectorySeedSource(path=str(seed_path), file_pattern="*.md"),
        selection_strategy=selection_strategy,
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="section_summary",
            expr="{{ file_name }} :: {{ section_header }}",
        )
    )
    return config_builder


def print_preview(
    *,
    data_designer: DataDesigner,
    title: str,
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
) -> None:
    """Run a preview and print the columns that matter for the walkthrough."""

    print(title)
    preview = data_designer.preview(config_builder, num_records=num_records)
    print(
        preview.dataset[
            [
                "relative_path",
                "section_index",
                "section_header",
                "section_summary",
            ]
        ].to_string(index=False)
    )
    print()


def main() -> None:
    """Build sample input files and print previews with and without selection."""

    with TemporaryDirectory(prefix="markdown-seed-reader-") as temp_dir:
        seed_dir = Path(temp_dir) / "sample_markdown"
        seed_dir.mkdir()
        create_sample_markdown_files(seed_dir)

        data_designer = DataDesigner(seed_readers=[MarkdownSectionDirectorySeedReader()])

        print_preview(
            data_designer=data_designer,
            title="Full preview across all markdown files",
            config_builder=build_config(seed_path=seed_dir),
            num_records=4,
        )
        print_preview(
            data_designer=data_designer,
            title="Manifest-based selection of only the second matched file",
            config_builder=build_config(
                seed_path=seed_dir,
                selection_strategy=IndexRange(start=1, end=1),
            ),
            num_records=2,
        )


if __name__ == "__main__":
    main()
