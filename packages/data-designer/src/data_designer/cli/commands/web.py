# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer


def review_command(
    data_file: str = typer.Argument(
        help="Path to a parquet or JSON file containing preview results to review.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind the server to."),
    port: int = typer.Option(8765, "--port", "-p", help="Port to bind the server to."),
    open_browser: bool = typer.Option(False, "--open", help="Auto-open the review UI in a browser."),
) -> None:
    """Launch the dataset review UI for inspecting and annotating preview results.

    Load a parquet or JSON file from a Data Designer preview run and open
    an interactive browser-based UI for reviewing records, rating them
    good/bad, and adding notes.

    Examples:
        # Review a parquet file
        data-designer review preview.parquet

        # Review and auto-open browser
        data-designer review preview.parquet --open

        # Custom port
        data-designer review preview.parquet --port 9000
    """
    from data_designer.web.server import run_server

    run_server(data_file=data_file, host=host, port=port, open_browser=open_browser)
