# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI server for the dataset review UI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from data_designer.web.api import init_session
from data_designer.web.api import router as api_router
from data_designer.web.session import ReviewSession

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(*, data_file: str) -> FastAPI:
    app = FastAPI(
        title="Data Designer Review",
        description="Dataset review UI for NeMo Data Designer",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session = ReviewSession(data_file=Path(data_file))
    init_session(session)

    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    async def redirect_to_ui():
        return RedirectResponse(url="/ui/")

    if STATIC_DIR.is_dir():
        app.mount("/ui/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
    else:
        logger.warning(f"Static assets not found at {STATIC_DIR}. Run 'npm run build' in packages/data-designer-web/")

    return app


def run_server(*, data_file: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    import uvicorn

    import data_designer.web.server as _self
    _self._startup_data_file = data_file

    if open_browser:
        import threading
        import time
        import webbrowser

        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    logger.info(f"Starting review UI at http://{host}:{port}")
    uvicorn.run(
        "data_designer.web.server:_create_app_from_startup",
        factory=True,
        host=host,
        port=port,
        log_level="info",
    )


_startup_data_file: str = ""


def _create_app_from_startup() -> FastAPI:
    return create_app(data_file=_startup_data_file)
