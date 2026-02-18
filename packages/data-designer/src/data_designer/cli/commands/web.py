# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer


def web_command(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind the server to."),
    port: int = typer.Option(8765, "--port", "-p", help="Port to bind the server to."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
) -> None:
    """Launch the Data Designer web config builder UI.

    Opens a browser-based interface for visually building, editing, previewing,
    and exporting Data Designer configuration files.

    Examples:
        # Start with default settings
        data-designer web

        # Custom host and port
        data-designer web --host 0.0.0.0 --port 9000

        # Enable auto-reload for backend development
        data-designer web --reload
    """
    from data_designer.web.server import run_server

    run_server(host=host, port=port, reload=reload)
