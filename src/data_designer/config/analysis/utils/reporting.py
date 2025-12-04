# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Column, Table
from rich.text import Text

from data_designer.config.analysis.column_statistics import CategoricalHistogramData
from data_designer.config.analysis.utils.errors import AnalysisReportError
from data_designer.config.column_types import COLUMN_TYPE_EMOJI_MAP, DataDesignerColumnType, get_column_display_order
from data_designer.config.utils.visualization import (
    ColorPalette,
    convert_to_row_element,
    create_rich_histogram_table,
    pad_console_element,
)

if TYPE_CHECKING:
    from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults

HEADER_STYLE = "dim"
RULE_STYLE = f"bold {ColorPalette.NVIDIA_GREEN.value}"
ACCENT_STYLE = f"bold {ColorPalette.BLUE.value}"
TITLE_STYLE = f"bold {ColorPalette.NVIDIA_GREEN.value}"
HIST_NAME_STYLE = f"bold {ColorPalette.BLUE.value}"
HIST_VALUE_STYLE = f"dim {ColorPalette.BLUE.value}"


class ReportSection(str, Enum):
    OVERVIEW = "overview"
    COLUMN_PROFILERS = "column_profilers"


DEFAULT_INCLUDE_SECTIONS = [
    ReportSection.OVERVIEW,
    ReportSection.COLUMN_PROFILERS,
] + get_column_display_order()


def generate_analysis_report(
    analysis: DatasetProfilerResults,
    save_path: Optional[Union[str, Path]] = None,
    include_sections: Optional[list[Union[ReportSection, DataDesignerColumnType]]] = None,
) -> None:
    """Generate an analysis report for dataset profiling results.

    This function creates a rich-formatted report that displays dataset overview statistics
    and detailed column statistics organized by column type. The report includes visual
    elements like tables, rules, and panels to present the analysis results in an
    easy-to-read format.

    Args:
        analysis: The DatasetProfilerResults object containing the analysis data to report on.
        save_path: Optional path to save the report. If provided, the report will be saved
                  as either HTML (.html) or SVG (.svg) format. If None, the report will
                  only be displayed in the console.
        include_sections: Optional list of sections to include in the report. Choices are
                  any Data Designer column type, "overview" (the dataset overview section),
                  and "column_profilers" (all column profilers in one section). If None,
                  all sections will be included.

    Raises:
        AnalysisReportError: If save_path is provided but doesn't have a .html or .svg extension.
    """
    render_list = []
    table_kws = dict(show_lines=True, expand=True, title_style=TITLE_STYLE)
    include_sections = include_sections or DEFAULT_INCLUDE_SECTIONS

    title = Rule(title="🎨 Data Designer Dataset Profile", style=RULE_STYLE, end="\n\n")

    render_list.append(title)

    if ReportSection.OVERVIEW in include_sections:
        table = Table(title="Dataset Overview", **table_kws)
        table.add_column("number of records", header_style=HEADER_STYLE)
        table.add_column("number of columns", header_style=HEADER_STYLE)
        table.add_column("percent complete records", header_style=HEADER_STYLE)

        table.add_row(
            f"{analysis.num_records:,}",
            f"{len(analysis.column_statistics):,}",
            f"{analysis.percent_complete:.1f}%",
        )

        render_list.append(pad_console_element(table, (1, 0, 1, 0)))

    displayed_column_types = set()
    for column_type in analysis.column_types:
        if column_type not in include_sections:
            continue

        displayed_column_types.add(column_type)
        column_label = column_type.replace("_", " ").title().replace("Llm", "LLM")
        table = Table(
            title=f"{COLUMN_TYPE_EMOJI_MAP[column_type]} {column_label} Columns",
            **table_kws,
        )

        column_stats_list = analysis.get_column_statistics_by_type(column_type)
        for col in list(column_stats_list[0].create_report_row_data()):
            if col == "column name":
                table.add_column(col, header_style=HEADER_STYLE)
            else:
                table.add_column(col, justify="right", header_style=HEADER_STYLE)

        for stats in column_stats_list:
            table.add_row(*[convert_to_row_element(elem) for elem in stats.create_report_row_data().values()])

        render_list.append(pad_console_element(table, (1, 0, 1, 0)))

    if ReportSection.COLUMN_PROFILERS in include_sections:
        for profile in analysis.column_profiles or []:
            render_list.append(pad_console_element(profile.create_report_section()))

    if any("llm" in col_type for col_type in displayed_column_types):
        footnotes_text = (
            "1. All token statistics are based on a sample of max(1000, len(dataset)) records.\n"
            "2. Tokens are calculated using tiktoken's cl100k_base tokenizer."
        )

        render_list.append(
            pad_console_element(
                Panel(
                    Text.from_markup(footnotes_text.strip()),
                    title="Table Notes",
                    border_style="dim",
                    padding=(1, 2),
                )
            )
        )

    render_list.append(Rule(style=RULE_STYLE))

    console = Console(record=save_path is not None)
    console.print(Group(*render_list), markup=False)

    if save_path is not None:
        save_path = str(save_path)
        if save_path.endswith(".html"):
            console.save_html(save_path)
        elif save_path.endswith(".svg"):
            console.save_svg(save_path, title="")
        else:
            raise AnalysisReportError(
                f"🛑 The extension of the save path must be either .html or .svg. You provided {save_path}."
            )


def create_judge_score_summary_table(
    score_name: str,
    histogram: CategoricalHistogramData,
    summary: str,
    accent_style: str = ACCENT_STYLE,
    summary_border_style: str = "dim",
) -> Table:
    layout = Table.grid(Column(), Column(), expand=True, padding=(0, 2))

    histogram_table = create_rich_histogram_table(
        {str(s): c for s, c in zip(histogram.categories, histogram.counts)},
        ("score", "count"),
        name_style=HIST_NAME_STYLE,
        value_style=HIST_VALUE_STYLE,
    )

    summary_panel = Panel(
        Text(summary, justify="left"),
        title=(
            f"Score Summary: [not {summary_border_style}][{accent_style}]"
            f"{score_name.upper()}[/{accent_style}][/not {summary_border_style}]"
        ),
        border_style=summary_border_style,
    )

    layout.add_row(
        Align(summary_panel, vertical="top"),
        Align(histogram_table, vertical="top"),
    )

    return layout
