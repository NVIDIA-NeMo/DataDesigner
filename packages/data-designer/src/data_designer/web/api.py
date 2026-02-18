# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""REST API routes for the Data Designer web UI."""

from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from data_designer.web.schema import (
    get_column_schemas,
    get_constraint_schemas,
    get_enum_values,
    get_full_config_schema,
    get_model_config_schema,
    get_processor_schemas,
    get_seed_config_schema,
    get_tool_config_schema,
)
from data_designer.web.session import ConfigSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

session = ConfigSession()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ColumnRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    column_type: str
    name: str


class ModelConfigRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    alias: str
    model: str


class LoadConfigRequest(BaseModel):
    config: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Schema endpoints
# ---------------------------------------------------------------------------

@router.get("/schema/columns", tags=["schema"])
async def schema_columns() -> dict[str, Any]:
    """Return JSON schemas for all column types."""
    return get_column_schemas()


@router.get("/schema/models", tags=["schema"])
async def schema_models() -> dict[str, Any]:
    return get_model_config_schema()


@router.get("/schema/seed", tags=["schema"])
async def schema_seed() -> dict[str, Any]:
    return get_seed_config_schema()


@router.get("/schema/tools", tags=["schema"])
async def schema_tools() -> dict[str, Any]:
    return get_tool_config_schema()


@router.get("/schema/constraints", tags=["schema"])
async def schema_constraints() -> dict[str, Any]:
    return get_constraint_schemas()


@router.get("/schema/processors", tags=["schema"])
async def schema_processors() -> dict[str, Any]:
    return get_processor_schemas()


@router.get("/schema/full", tags=["schema"])
async def schema_full() -> dict[str, Any]:
    return get_full_config_schema()


@router.get("/schema/enums", tags=["schema"])
async def schema_enums() -> dict[str, list[str]]:
    return get_enum_values()


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@router.get("/config", tags=["config"])
async def get_config() -> dict[str, Any]:
    """Return the current builder config as JSON."""
    try:
        return session.get_config_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/config", tags=["config"])
async def load_config(req: LoadConfigRequest) -> dict[str, Any]:
    """Replace the current config from a JSON payload."""
    try:
        session.load_config(req.config)
        return session.get_config_dict()
    except Exception as e:
        logger.error(f"Failed to load config: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/config", tags=["config"])
async def reset_config() -> dict[str, str]:
    """Reset to an empty config."""
    session.reset()
    return {"status": "ok"}


@router.get("/config/export", tags=["config"])
async def export_config(fmt: str = Query("yaml", alias="format")) -> dict[str, str]:
    """Export config as YAML or JSON string."""
    try:
        if fmt == "yaml":
            return {"format": "yaml", "content": session.get_config_yaml()}
        return {"format": "json", "content": session.get_config_json()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Column endpoints
# ---------------------------------------------------------------------------

@router.get("/config/columns", tags=["columns"])
async def list_columns() -> list[dict[str, Any]]:
    return session.list_columns()


@router.post("/config/columns", tags=["columns"])
async def add_column(body: dict[str, Any]) -> dict[str, Any]:
    try:
        return session.add_column(body)
    except Exception as e:
        logger.error(f"Failed to add column: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=str(e))


@router.put("/config/columns/{name}", tags=["columns"])
async def update_column(name: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        return session.update_column(name, body)
    except Exception as e:
        logger.error(f"Failed to update column: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/config/columns/{name}", tags=["columns"])
async def delete_column(name: str) -> dict[str, str]:
    try:
        session.delete_column(name)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Model config endpoints
# ---------------------------------------------------------------------------

@router.get("/config/models", tags=["models"])
async def list_models() -> list[dict[str, Any]]:
    return session.list_models()


@router.post("/config/models", tags=["models"])
async def add_model(body: dict[str, Any]) -> dict[str, Any]:
    try:
        return session.add_model(body)
    except Exception as e:
        logger.error(f"Failed to add model: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/config/models/{alias}", tags=["models"])
async def delete_model(alias: str) -> dict[str, str]:
    try:
        session.delete_model(alias)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@router.get("/references", tags=["utils"])
async def get_references() -> list[str]:
    """Column names available for Jinja2 template references."""
    return session.get_allowed_references()


@router.get("/health", tags=["utils"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
