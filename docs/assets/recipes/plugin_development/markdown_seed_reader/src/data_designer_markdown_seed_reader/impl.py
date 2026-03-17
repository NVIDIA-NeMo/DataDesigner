# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import FileSystemSeedReader, SeedReaderFileSystemContext
from data_designer_markdown_seed_reader.config import MarkdownSectionSeedSource

_ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")


class MarkdownSectionSeedReader(FileSystemSeedReader[MarkdownSectionSeedSource]):
    output_columns: ClassVar[list[str]] = [
        "relative_path",
        "file_name",
        "section_index",
        "section_header",
        "section_content",
    ]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
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
        relative_path = str(manifest_row["relative_path"])
        file_name = str(manifest_row["file_name"])
        with context.fs.open(relative_path, "r", encoding="utf-8") as handle:
            markdown_text = handle.read()

        sections = _extract_markdown_sections(markdown_text=markdown_text, fallback_header=file_name)
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


def _extract_markdown_sections(*, markdown_text: str, fallback_header: str) -> list[tuple[str, str]]:
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
