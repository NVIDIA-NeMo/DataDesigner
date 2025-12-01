# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.visualization import WithRecordSamplerMixin
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.processing.processors.registry import ProcessorRegistry


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

    def write_processors_outputs_to_disk(
        self,
        processors: list[str],
        output_folder: Path | str,
    ) -> None:
        """Write collected artifacts from each processor to disk.

        Args:
            processors (list[str]): List of processor names to collect artifacts from.
            output_folder (Path | str): Path to the output folder.

        Returns:
            None
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        processors = set(processors)
        for processor_config in self._config_builder.get_processor_configs():
            if processor_config.name not in processors:
                continue

            ProcessorClass = ProcessorRegistry.get_for_config_type(type(processor_config))
            ProcessorClass.write_outputs_to_disk(
                processor_config=processor_config,
                artifacts_path=self.artifact_storage.processors_outputs_path / processor_config.name,
                output_path=output_folder / processor_config.name,
            )
