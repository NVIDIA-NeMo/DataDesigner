# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory session management for the web UI.

Holds a single DataDesignerConfigBuilder instance that the API endpoints
mutate.  For the web UI use case a single global config session is sufficient
-- the UI is a local dev tool, not a multi-tenant service.
"""

from __future__ import annotations

import logging
from typing import Any

from data_designer.config.column_types import ColumnConfigT, get_column_config_from_kwargs
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.config.utils.io_helpers import serialize_data, smart_load_yaml

logger = logging.getLogger(__name__)


class ConfigSession:
    """Wraps a DataDesignerConfigBuilder with helper methods for the web API."""

    def __init__(self) -> None:
        self._builder: DataDesignerConfigBuilder | None = None

    @property
    def builder(self) -> DataDesignerConfigBuilder:
        if self._builder is None:
            self._builder = DataDesignerConfigBuilder(model_configs=[])
        return self._builder

    @builder.setter
    def builder(self, value: DataDesignerConfigBuilder) -> None:
        self._builder = value

    @property
    def is_initialized(self) -> bool:
        return self._builder is not None

    def reset(self) -> None:
        self._builder = None

    # -- Config I/O ---------------------------------------------------------

    def load_config(self, config_data: dict | str) -> None:
        """Load a config from a dict (parsed JSON/YAML) or raw YAML/JSON string."""
        if isinstance(config_data, str):
            config_data = smart_load_yaml(config_data)
        self._builder = DataDesignerConfigBuilder.from_config(config_data)

    def get_config_dict(self) -> dict[str, Any]:
        """Return the current config as a JSON-serialisable dict."""
        return self.builder.get_builder_config().to_dict()

    def get_config_yaml(self) -> str:
        return self.builder.get_builder_config().to_yaml() or ""

    def get_config_json(self) -> str:
        return serialize_data(self.get_config_dict(), indent=2)

    # -- Column CRUD --------------------------------------------------------

    def list_columns(self) -> list[dict[str, Any]]:
        columns = self.builder.get_column_configs()
        result = []
        for col in columns:
            col_dict = col.model_dump(mode="json")
            col_dict["_column_type"] = col.column_type
            result.append(col_dict)
        return result

    def add_column(self, column_data: dict[str, Any]) -> dict[str, Any]:
        """Add a column from a raw dict. Returns the created column as a dict."""
        column_type = column_data.pop("column_type", None)
        name = column_data.pop("name", None)
        if column_type is None or name is None:
            raise ValueError("Both 'column_type' and 'name' are required")
        column_data["column_type"] = column_type
        column_data["name"] = name
        col = _parse_column_config(column_data)
        self.builder.add_column(col)
        return col.model_dump(mode="json")

    def update_column(self, name: str, column_data: dict[str, Any]) -> dict[str, Any]:
        self.builder.delete_column(name)
        column_data["name"] = column_data.get("name", name)
        return self.add_column(column_data)

    def delete_column(self, name: str) -> None:
        self.builder.delete_column(name)

    # -- Model CRUD ---------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        return [mc.model_dump(mode="json") for mc in self.builder.model_configs]

    def add_model(self, model_data: dict[str, Any]) -> dict[str, Any]:
        mc = ModelConfig.model_validate(model_data)
        self.builder.add_model_config(mc)
        return mc.model_dump(mode="json")

    def delete_model(self, alias: str) -> None:
        self.builder.delete_model_config(alias)

    # -- Allowed references -------------------------------------------------

    def get_allowed_references(self) -> list[str]:
        return self.builder.allowed_references


def _parse_column_config(data: dict[str, Any]) -> ColumnConfigT:
    """Parse a raw dict into a typed column config using the discriminated union."""
    column_type = data.get("column_type")
    name = data.get("name")

    from data_designer.config.column_types import _COLUMN_TYPE_CONFIG_CLS_MAP, DataDesignerColumnType
    from data_designer.config.utils.type_helpers import resolve_string_enum

    col_type_enum = resolve_string_enum(column_type, DataDesignerColumnType)
    config_cls = _COLUMN_TYPE_CONFIG_CLS_MAP.get(col_type_enum)
    if config_cls is None:
        raise ValueError(f"Unknown column_type: {column_type}")

    if col_type_enum == DataDesignerColumnType.SAMPLER:
        return get_column_config_from_kwargs(name=name, column_type=col_type_enum, **{k: v for k, v in data.items() if k not in ("name", "column_type")})

    return config_cls.model_validate(data)
