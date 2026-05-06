# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from data_designer.config.config_builder import DataDesignerConfigBuilder


@dataclass(frozen=True)
class DataDesignerRecipe:
    """A reusable Data Designer pipeline composition.

    Recipe packages register instances of this class through the
    ``data_designer.recipes`` entry point group. The Data Designer CLI owns the
    generic execution flow; recipe packages own config validation and the
    construction of a :class:`DataDesignerConfigBuilder`.

    Attributes:
        name: Stable recipe name used by ``data-designer run-recipe``.
        description: Human-readable summary shown by ``data-designer recipes``.
        config_model: Pydantic model class used to validate recipe config files.
        build_config: Callable that converts a validated recipe config into a
            :class:`DataDesignerConfigBuilder`.
        postprocess: Optional callback invoked after ``create`` runs. This is
            intended for exports or recipe-specific artifacts, not for adding
            generation columns.
    """

    name: str
    description: str
    config_model: type[BaseModel]
    build_config: Callable[[BaseModel], DataDesignerConfigBuilder]
    postprocess: Callable[[Any, BaseModel], None] | None = None
