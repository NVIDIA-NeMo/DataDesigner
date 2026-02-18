# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""REST API routes for the Data Designer web UI.

Execution-oriented: load configs from files, validate, preview, create,
and inspect results with traces.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from data_designer.web.session import ExecutionSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_session: ExecutionSession | None = None


def init_session(session: ExecutionSession) -> None:
    """Called by server.py to inject the session after app creation."""
    global _session
    _session = session


def _get_session() -> ExecutionSession:
    if _session is None:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return _session


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoadConfigRequest(BaseModel):
    path: str


class PreviewRequest(BaseModel):
    num_records: int = 10
    debug_mode: bool = False


class CreateRequest(BaseModel):
    num_records: int = 100
    dataset_name: str = "dataset"
    artifact_path: str | None = None


# ---------------------------------------------------------------------------
# Config discovery & loading
# ---------------------------------------------------------------------------

@router.get("/configs", tags=["config"])
async def list_configs() -> list[dict[str, Any]]:
    """List config files found in the working directory."""
    return _get_session().list_configs()


@router.post("/config/load", tags=["config"])
async def load_config(req: LoadConfigRequest) -> dict[str, Any]:
    """Load a config file by path."""
    try:
        return _get_session().load_config(req.path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/config", tags=["config"])
async def get_config() -> dict[str, Any]:
    """Return the loaded config as JSON."""
    s = _get_session()
    if not s.is_loaded:
        return {}
    return s.get_config_dict()


@router.get("/config/yaml", tags=["config"])
async def get_config_yaml() -> dict[str, str]:
    """Return the loaded config as YAML text."""
    return {"content": _get_session().get_config_yaml()}


@router.get("/config/export", tags=["config"])
async def export_config(fmt: str = Query("yaml", alias="format")) -> dict[str, str]:
    """Export config as YAML or JSON string."""
    s = _get_session()
    if fmt == "yaml":
        return {"format": "yaml", "content": s.get_config_yaml()}
    return {"format": "json", "content": s.get_config_json()}


@router.get("/config/columns", tags=["config"])
async def list_columns() -> list[dict[str, Any]]:
    """List columns from the loaded config (read-only)."""
    return _get_session().list_columns()


@router.get("/config/models", tags=["config"])
async def list_models() -> list[dict[str, Any]]:
    """List model configs from the loaded config (read-only)."""
    return _get_session().list_models()


@router.get("/config/info", tags=["config"])
async def config_info() -> dict[str, Any]:
    """Summary info about the loaded config."""
    s = _get_session()
    return {
        "loaded": s.is_loaded,
        "path": s.config_path,
        "columns": s.list_columns(),
        "models": s.list_models(),
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@router.post("/config/validate", tags=["execution"])
async def validate_config() -> dict[str, Any]:
    """Validate the loaded config. Runs in a thread to avoid blocking the event loop."""
    s = _get_session()
    if not s.is_loaded:
        raise HTTPException(status_code=400, detail="No config loaded")
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, s.validate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/preview", tags=["execution"])
async def run_preview(req: PreviewRequest) -> dict[str, Any]:
    """Start a preview run (async). Poll /api/status for progress."""
    s = _get_session()
    if not s.is_loaded:
        raise HTTPException(status_code=400, detail="No config loaded")
    try:
        s.run_preview(num_records=req.num_records, debug_mode=req.debug_mode)
        return {"status": "started", "type": "preview"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/create", tags=["execution"])
async def run_create(req: CreateRequest) -> dict[str, Any]:
    """Start a full create run (async). Poll /api/status for progress."""
    s = _get_session()
    if not s.is_loaded:
        raise HTTPException(status_code=400, detail="No config loaded")
    try:
        s.run_create(
            num_records=req.num_records,
            dataset_name=req.dataset_name,
            artifact_path=req.artifact_path,
        )
        return {"status": "started", "type": "create"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status", tags=["execution"])
async def execution_status() -> dict[str, Any]:
    """Get current execution state."""
    return _get_session().execution_status


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@router.get("/preview/results", tags=["results"])
async def preview_results() -> dict[str, Any]:
    """Get cached preview results (dataset rows, columns, analysis)."""
    return _get_session().get_preview_results()


@router.get("/preview/traces/{row}/{column}", tags=["results"])
async def preview_trace(row: int, column: str) -> list[dict[str, Any]]:
    """Get trace data for a specific row and column."""
    return _get_session().get_preview_trace(row, column)


@router.get("/create/results", tags=["results"])
async def create_results() -> dict[str, Any]:
    """Get create run results (artifact path, record count)."""
    return _get_session().get_create_result()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@router.get("/health", tags=["utils"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
