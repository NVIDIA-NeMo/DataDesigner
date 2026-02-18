# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory and server entry point for the Data Designer web UI.

Mirrors the ADK-Python pattern: compiled frontend assets are served as static
files alongside a JSON API that wraps the DataDesignerConfigBuilder.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from data_designer.web.api import router as api_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(*, allow_origins: list[str] | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        allow_origins: CORS allowed origins. Defaults to permissive localhost
            origins suitable for local development.
    """
    app = FastAPI(
        title="Data Designer Web UI",
        description="Visual config builder for NeMo Data Designer",
    )

    if allow_origins is None:
        allow_origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    async def redirect_to_ui():
        return RedirectResponse(url="/ui/")

    if STATIC_DIR.is_dir():
        app.mount(
            "/ui/",
            StaticFiles(directory=str(STATIC_DIR), html=True),
            name="ui",
        )
    else:
        logger.warning(
            f"Static assets directory not found at {STATIC_DIR}. "
            "The web UI will not be available. "
            "Run 'npm run build' in packages/data-designer-web/ to compile the frontend."
        )

    return app


def run_server(*, host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Start the uvicorn server."""
    import uvicorn

    logger.info(f"Starting Data Designer web UI at http://{host}:{port}")
    uvicorn.run(
        "data_designer.web.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
