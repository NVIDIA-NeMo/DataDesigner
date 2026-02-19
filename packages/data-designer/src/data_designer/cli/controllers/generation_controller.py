# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer

from data_designer.cli.ui import console, print_error, print_header, print_success, wait_for_navigation_key
from data_designer.cli.utils.config_loader import ConfigLoadError, load_config_builder
from data_designer.cli.utils.sample_records_pager import PAGER_FILENAME, create_sample_records_pager
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.constants import DEFAULT_DISPLAY_WIDTH
from data_designer.interface import DataDesigner
from data_designer.logging import LOG_INDENT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from data_designer.config.config_builder import DataDesignerConfigBuilder
    from data_designer.config.preview_results import PreviewResults


class GenerationController:
    """Controller for dataset generation workflows (preview, validate, create)."""

    def run_preview(
        self,
        config_source: str,
        num_records: int,
        non_interactive: bool,
        save_results: bool = False,
        artifact_path: str | None = None,
        theme: Literal["dark", "light"] = "dark",
        display_width: int = DEFAULT_DISPLAY_WIDTH,
    ) -> None:
        """Load config, generate a preview dataset, and display the results.

        Args:
            config_source: Path to a config file or Python module.
            num_records: Number of records to generate.
            non_interactive: If True, display all records at once instead of browsing.
            save_results: If True, save all preview artifacts to the artifact path.
            artifact_path: Directory to save results in, or None for ./artifacts.
            theme: Color theme for HTML output (dark or light).
            display_width: Maximum width of the rendered record output in characters.
        """
        config_builder = self._load_config(config_source)

        print_header("Data Designer Preview")
        console.print(f"  Config: [bold]{config_source}[/bold]")
        console.print(f"  Records: [bold]{num_records}[/bold]")
        console.print()

        try:
            data_designer = DataDesigner()
            results = data_designer.preview(config_builder, num_records=num_records)
        except Exception as e:
            print_error(f"Preview generation failed: {e}")
            raise typer.Exit(code=1)

        if results.dataset is None or len(results.dataset) == 0:
            print_error("No records were generated.")
            raise typer.Exit(code=1)

        total = len(results.dataset)

        if save_results:
            self._save_preview_results(results, total, artifact_path, theme, display_width)
        else:
            use_interactive = not non_interactive and sys.stdin.isatty() and sys.stdout.isatty() and total > 1
            if use_interactive:
                self._browse_records_interactively(results, total, display_width)
            else:
                self._display_all_records(results, total, display_width)

            if results.analysis is not None:
                console.print()
                results.analysis.to_report()

        console.print()
        print_success(f"Preview complete — {total} record(s) generated")

    def run_validate(self, config_source: str) -> None:
        """Load config and validate it against the engine.

        Args:
            config_source: Path to a config file or Python module.
        """
        config_builder = self._load_config(config_source)

        print_header("Data Designer Validate")
        console.print(f"  Config: [bold]{config_source}[/bold]")
        console.print()

        try:
            data_designer = DataDesigner()
            data_designer.validate(config_builder)
        except InvalidConfigError as e:
            print_error(f"Configuration is invalid: {e}")
            raise typer.Exit(code=1)
        except Exception as e:
            print_error(f"Validation failed: {e}")
            raise typer.Exit(code=1)

        print_success("Configuration is valid")

    def run_create(
        self,
        config_source: str,
        num_records: int,
        dataset_name: str,
        artifact_path: str | None,
    ) -> None:
        """Load config, create a full dataset, and save results to disk.

        Args:
            config_source: Path to a config file or Python module.
            num_records: Number of records to generate.
            dataset_name: Name for the generated dataset folder.
            artifact_path: Path where generated artifacts will be stored, or None for default.
        """
        config_builder = self._load_config(config_source)

        resolved_artifact_path = Path(artifact_path) if artifact_path else Path.cwd() / "artifacts"

        print_header("Data Designer Create")
        console.print(f"  Config: [bold]{config_source}[/bold]")
        console.print(f"  Records: [bold]{num_records}[/bold]")
        console.print(f"  Dataset name: [bold]{dataset_name}[/bold]")
        console.print(f"  Artifact path: [bold]{resolved_artifact_path}[/bold]")
        console.print()

        try:
            data_designer = DataDesigner(artifact_path=resolved_artifact_path)
            results = data_designer.create(
                config_builder,
                num_records=num_records,
                dataset_name=dataset_name,
            )
        except Exception as e:
            print_error(f"Dataset creation failed: {e}")
            raise typer.Exit(code=1)

        dataset = results.load_dataset()

        analysis = results.load_analysis()
        if analysis is not None:
            console.print()
            analysis.to_report()

        console.print()
        print_success(f"Dataset created — {len(dataset)} record(s) generated")
        console.print(f"  Artifacts saved to: [bold]{results.artifact_storage.base_dataset_path}[/bold]")
        console.print()

    def _load_config(self, config_source: str) -> DataDesignerConfigBuilder:
        """Load a config builder from the given source, exiting on failure.

        Args:
            config_source: Path to a config file or Python module.

        Returns:
            A DataDesignerConfigBuilder instance.

        Raises:
            typer.Exit: If the config cannot be loaded.
        """
        try:
            return load_config_builder(config_source)
        except ConfigLoadError as e:
            print_error(str(e))
            raise typer.Exit(code=1)

    def _save_preview_results(
        self,
        results: PreviewResults,
        total: int,
        artifact_path: str | None,
        theme: Literal["dark", "light"],
        display_width: int,
    ) -> None:
        """Save all preview artifacts to disk without displaying in the terminal."""
        try:
            resolved_artifact_path = Path(artifact_path) if artifact_path else Path.cwd() / "artifacts"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = resolved_artifact_path / f"preview_results_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)

            if results.analysis is not None:
                results.analysis.to_report(save_path=results_dir / "report.html")

            results.dataset.to_parquet(results_dir / "dataset.parquet")

            sample_records_dir = results_dir / "sample_records"
            sample_records_dir.mkdir(parents=True, exist_ok=True)
            for i in range(total):
                results.display_sample_record(
                    index=i,
                    save_path=sample_records_dir / f"record_{i}.html",
                    theme=theme,
                    display_width=display_width,
                )
            create_sample_records_pager(
                sample_records_dir=sample_records_dir,
                num_records=total,
                num_columns=len(results.dataset.columns),
                theme=theme,
            )

            logger.info(f"{LOG_INDENT}Results path: {results_dir}")
            logger.info(f"{LOG_INDENT}Browser path: {sample_records_dir / PAGER_FILENAME}")
        except OSError as e:
            print_error(f"Failed to save preview results: {e}")
            raise typer.Exit(code=1)

    def _display_record_with_header(
        self, results: PreviewResults, index: int, total: int, display_width: int = DEFAULT_DISPLAY_WIDTH
    ) -> None:
        """Display a single record with a record number header."""
        console.print(f"  [bold]Record {index + 1} of {total}[/bold]")
        results.display_sample_record(index=index, display_width=display_width)

    def _browse_records_interactively(
        self, results: PreviewResults, total: int, display_width: int = DEFAULT_DISPLAY_WIDTH
    ) -> None:
        """Interactively browse records with single-keypress navigation.

        Shows the first record immediately, then waits for navigation keys.
        Controls: n/enter=next, p=previous, q/Escape/Ctrl+C=quit.
        Navigation wraps around at both ends.
        """
        current_index = 0
        self._display_record_with_header(results, current_index, total, display_width)

        while True:
            console.print()
            action = wait_for_navigation_key()

            if action == "q":
                console.print("  [dim]Done browsing.[/dim]")
                break
            if action == "p":
                current_index = (current_index - 1) % total
            else:
                current_index = (current_index + 1) % total

            self._display_record_with_header(results, current_index, total, display_width)

    def _display_all_records(
        self, results: PreviewResults, total: int, display_width: int = DEFAULT_DISPLAY_WIDTH
    ) -> None:
        """Display all records without interactive prompts."""
        for i in range(total):
            self._display_record_with_header(results, i, total, display_width)
