# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

from data_designer.cli.ui import console, print_error, print_header, print_info
from data_designer.config.utils.io_helpers import validate_dataset_file_path


class ReviewController:
    """Controller for dataset review workflow.

    Orchestrates the dataset review process by validating inputs and launching
    the Streamlit review UI.
    """

    def __init__(self, dataset_path: Path, port: int, host: str, reviewer: str):
        """Initialize controller with review parameters.

        Args:
            dataset_path: Path to dataset file to review
            port: Port for Streamlit server
            host: Host address to bind
            reviewer: Reviewer identifier
        """
        self.dataset_path = dataset_path
        self.port = port
        self.host = host
        self.reviewer = reviewer

    def run(self) -> None:
        """Main entry point for review workflow.

        Validates the dataset path and launches the Streamlit review UI.
        """
        print_header("Dataset Review UI")

        # Validate dataset path
        try:
            self.dataset_path = validate_dataset_file_path(self.dataset_path, should_exist=True)
        except Exception as e:
            print_error(f"Invalid dataset path: {e}")
            return

        print_info(f"Dataset: {self.dataset_path}")
        print_info(f"Server: http://{self.host}:{self.port}")
        print_info(f"Reviewer: {self.reviewer}")
        console.print()

        # Launch Streamlit app
        self._launch_streamlit_app()

    def _launch_streamlit_app(self) -> None:
        """Launch the Streamlit application via subprocess."""
        import data_designer.cli.ui.review_app as review_app_module

        app_path = Path(review_app_module.__file__)

        try:
            print_info("Starting Streamlit server...")
            console.print()

            # Launch Streamlit with subprocess
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                f"--server.port={self.port}",
                f"--server.address={self.host}",
                "--browser.gatherUsageStats=false",
                "--",  # Args after this go to the Streamlit app
                str(self.dataset_path),
                self.reviewer,
            ]

            subprocess.run(cmd, check=True)

        except subprocess.CalledProcessError as e:
            print_error(f"Failed to launch Streamlit: {e}")
        except KeyboardInterrupt:
            console.print()
            print_info("Review session ended by user")
        except Exception as e:
            print_error(f"Unexpected error: {e}")
