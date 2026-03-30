# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed_source import DirectorySeedSource
from data_designer.engine.resources.seed_reader import FileSystemSeedReader, SeedReaderFileSystemContext


class LineFanoutDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self, *, include_file_name: bool = False) -> None:
        self.include_file_name = include_file_name
        self.hydrated_relative_paths: list[str] = []
        self.output_columns = ["relative_path", "line_index", "line"]
        if include_file_name:
            self.output_columns.insert(1, "file_name")

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                **({"file_name": Path(relative_path).name} if self.include_file_name else {}),
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
        self.hydrated_relative_paths.append(relative_path)
        with context.fs.open(relative_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        return [
            {
                "relative_path": relative_path,
                **({"file_name": str(manifest_row["file_name"])} if self.include_file_name else {}),
                "line_index": line_index,
                "line": line,
            }
            for line_index, line in enumerate(lines)
        ]
