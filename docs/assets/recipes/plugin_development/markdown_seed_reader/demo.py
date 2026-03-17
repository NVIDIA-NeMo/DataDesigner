# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from data_designer_markdown_seed_reader.config import MarkdownSectionSeedSource

import data_designer.config as dd
from data_designer.config.seed import IndexRange
from data_designer.interface import DataDesigner


def build_config(
    *,
    seed_path: Path,
    selection_strategy: IndexRange | None = None,
) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(
        MarkdownSectionSeedSource(path=str(seed_path)),
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
    project_root = Path(__file__).parent
    seed_path = project_root / "sample_data"
    data_designer = DataDesigner()

    print_preview(
        data_designer=data_designer,
        title="Full preview across all markdown files",
        config_builder=build_config(seed_path=seed_path),
        num_records=4,
    )
    print_preview(
        data_designer=data_designer,
        title="Manifest-based selection of only the second matched file",
        config_builder=build_config(
            seed_path=seed_path,
            selection_strategy=IndexRange(start=1, end=1),
        ),
        num_records=2,
    )


if __name__ == "__main__":
    main()
