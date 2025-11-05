# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator

from ..columns import DataDesignerColumnType, get_column_display_order
from ..utils.constants import EPSILON
from ..utils.numerical_helpers import prepare_number_for_reporting
from .column_profilers import ColumnProfilerResultsT
from .column_statistics import ColumnStatisticsT
from .utils.reporting import ReportSection, generate_analysis_report


class DatasetProfilerResults(BaseModel):
    num_records: int
    target_num_records: int
    column_statistics: list[ColumnStatisticsT] = Field(..., min_length=1)
    side_effect_column_names: Optional[list[str]] = None
    column_profiles: Optional[list[ColumnProfilerResultsT]] = None

    @field_validator("num_records", "target_num_records", mode="before")
    def ensure_python_integers(cls, v: int) -> int:
        return prepare_number_for_reporting(v, int)

    @property
    def percent_complete(self) -> float:
        return 100 * self.num_records / (self.target_num_records + EPSILON)

    @cached_property
    def column_types(self) -> list[str]:
        display_order = get_column_display_order()
        return sorted(
            list(set([c.column_type for c in self.column_statistics])),
            key=lambda x: display_order.index(x) if x in display_order else len(display_order),
        )

    def get_column_statistics_by_type(self, column_type: DataDesignerColumnType) -> list[ColumnStatisticsT]:
        return [c for c in self.column_statistics if c.column_type == column_type]

    def to_report(
        self,
        save_path: Optional[Union[str, Path]] = None,
        include_sections: Optional[list[Union[ReportSection, DataDesignerColumnType]]] = None,
    ) -> None:
        """Generate and print an analysis report based on the dataset profiling results.

        Args:
            save_path: Optional path to save the report. If provided, the report will be saved
                  as either HTML (.html) or SVG (.svg) format. If None, the report will
                  only be displayed in the console.
            include_sections: Optional list of sections to include in the report. Choices are
                  any DataDesignerColumnType, "overview" (the dataset overview section),
                  and "column_profilers" (all column profilers in one section). If None,
                  all sections will be included.
        """
        generate_analysis_report(self, save_path, include_sections=include_sections)
