# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory and server entry point for the Data Designer web UI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from data_designer.web.api import init_session
from data_designer.web.api import router as api_router
from data_designer.web.session import ExecutionSession

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    config_path: str | None = None,
    config_dir: str | None = None,
    allow_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_path: Optional path to a config file to load on startup.
        config_dir: Directory to scan for config files. Defaults to cwd.
        allow_origins: CORS allowed origins.
    """
    app = FastAPI(
        title="Data Designer Web UI",
        description="Execution dashboard for NeMo Data Designer",
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

    resolved_dir = Path(config_dir) if config_dir else Path.cwd()
    resolved_path = Path(config_path) if config_path else None

    if resolved_path and not resolved_path.is_absolute():
        resolved_path = resolved_dir / resolved_path

    session = ExecutionSession(
        config_dir=resolved_dir,
        config_path=resolved_path if resolved_path and resolved_path.exists() else None,
    )
    init_session(session)

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


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    reload: bool = False,
    config_path: str | None = None,
    config_dir: str | None = None,
) -> None:
    """Start the uvicorn server."""
    import uvicorn

    # Store args so the factory can pick them up when uvicorn calls it
    import data_designer.web.server as _self
    _self._startup_config_path = config_path
    _self._startup_config_dir = config_dir

    logger.info(f"Starting Data Designer web UI at http://{host}:{port}")
    uvicorn.run(
        "data_designer.web.server:_create_app_from_startup",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# Module-level vars set by run_server before uvicorn calls the factory
_startup_config_path: str | None = None
_startup_config_dir: str | None = None


def _create_app_from_startup() -> FastAPI:
    """Factory called by uvicorn, reads module-level startup vars."""
    return create_app(
        config_path=_startup_config_path,
        config_dir=_startup_config_dir,
    )
