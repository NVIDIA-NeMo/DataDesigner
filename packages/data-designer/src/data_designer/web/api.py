# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""REST API for the dataset review UI."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data_designer.web.session import ReviewSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_session: ReviewSession | None = None


def init_session(session: ReviewSession) -> None:
    global _session
    _session = session


def _get_session() -> ReviewSession:
    if _session is None:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return _session


class AnnotationRequest(BaseModel):
    row: int
    rating: str | None = None
    note: str = ""
    column: str | None = None


# -- Session ---------------------------------------------------------------

@router.get("/session", tags=["session"])
async def get_session() -> dict[str, Any]:
    return _get_session().get_session_info()


@router.get("/session/rows", tags=["session"])
async def get_rows() -> list[dict[str, Any]]:
    return _get_session().get_rows()


@router.get("/session/traces/{row}/{column}", tags=["session"])
async def get_trace(row: int, column: str) -> list[dict[str, Any]]:
    return _get_session().get_trace(row, column)


@router.post("/session/reload", tags=["session"])
async def reload_session() -> dict[str, Any]:
    """Re-read data file from disk, clear annotations. Called by agent after writing new parquet."""
    try:
        _get_session().reload()
        return _get_session().get_session_info()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/session/finish", tags=["session"])
async def finish_review() -> dict[str, Any]:
    """Write final annotations file and mark session complete."""
    return _get_session().finish_review()


# -- Annotations -----------------------------------------------------------

@router.get("/annotations", tags=["annotations"])
async def get_annotations() -> dict[str, Any]:
    return _get_session().get_annotations()


@router.post("/annotations", tags=["annotations"])
async def annotate_row(req: AnnotationRequest) -> dict[str, str]:
    _get_session().annotate_row(req.row, req.rating, req.note, req.column)
    return {"status": "ok"}


@router.get("/annotations/summary", tags=["annotations"])
async def annotations_summary() -> dict[str, int]:
    return _get_session().get_annotations_summary()


# -- Health ----------------------------------------------------------------

@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
