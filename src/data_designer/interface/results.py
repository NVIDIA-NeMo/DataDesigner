# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.visualization import WithRecordSamplerMixin
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage


class DatasetCreationResults(WithRecordSamplerMixin):
    """Results container for a Data Designer dataset creation run.

    This class provides access to the generated dataset, profiling analysis, and
    visualization utilities. It is returned by the DataDesigner.create() method
    and implements ResultsProtocol of the DataDesigner interface.
    """

    def __init__(
        self,
        *,
        artifact_storage: ArtifactStorage,
        analysis: DatasetProfilerResults,
        config_builder: DataDesignerConfigBuilder,
    ):
        """Creates a new instance with results based on a dataset creation run.

        Args:
            artifact_storage: Storage manager for accessing generated artifacts.
            analysis: Profiling results for the generated dataset.
            config_builder: Configuration builder used to create the dataset.
        """
        self.artifact_storage = artifact_storage
        self._analysis = analysis
        self._config_builder = config_builder

    def load_analysis(self) -> DatasetProfilerResults:
        """Load the profiling analysis results for the generated dataset.

        Returns:
            DatasetProfilerResults containing statistical analysis and quality metrics
                for each column in the generated dataset.
        """
        return self._analysis

    def load_dataset(self) -> pd.DataFrame:
        """Load the generated dataset as a pandas DataFrame.

        Returns:
            A pandas DataFrame containing the full generated dataset.
        """
        return self.artifact_storage.load_dataset()

    def write_processor_outputs_to_disk(self, output_folder: Path | str, extension: Literal["jsonl", "csv"]) -> None:
        """Write the processor outputs to disk.

        Returns:
            None
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        for subfolder in self.artifact_storage.processors_outputs_path.iterdir():
            output_file_path = output_folder / f"{subfolder.name}.{extension}"
            with open(output_file_path, "w") as f:
                for file_path in subfolder.glob("*.parquet"):
                    # TODO: faster way to convert than reading and writing row by row?
                    dataframe = pd.read_parquet(file_path)
                    for _, row in dataframe.iterrows():
                        f.write(row["formatted_output"].replace("\n", "\\n") + "\n")
