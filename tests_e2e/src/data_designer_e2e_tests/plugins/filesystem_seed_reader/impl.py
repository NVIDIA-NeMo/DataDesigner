# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import FileSystemSeedReader, SeedReaderFileSystemContext
from data_designer_e2e_tests.plugins.filesystem_seed_reader.config import DemoFileSystemSeedSource


class DemoFileSystemSeedReader(FileSystemSeedReader[DemoFileSystemSeedSource]):
    output_columns = ["relative_path", "file_name", "prefixed_content"]

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
    ) -> dict[str, str]:
        relative_path = str(manifest_row["relative_path"])
        with context.fs.open(relative_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
        return {
            "relative_path": relative_path,
            "file_name": str(manifest_row["file_name"]),
            "prefixed_content": f"{self.source.prefix}:{content}",
        }
