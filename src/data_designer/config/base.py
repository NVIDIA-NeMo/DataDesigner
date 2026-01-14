# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from data_designer.config.utils.io_helpers import serialize_data


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        json_schema_mode_override="validation",
    )


class ExportableConfigBase(ConfigBase):
    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            A dictionary representation of the configuration using JSON-compatible
            serialization.
        """
        return self.model_dump(mode="json")

    def to_yaml(self, path: str | Path | None = None, *, indent: int | None = 2, **kwargs) -> str | None:
        """Convert the configuration to a YAML string or file.

        Args:
            path: Optional file path to write the YAML to. If None, returns the
                YAML string instead of writing to file.
            indent: Number of spaces for YAML indentation. Defaults to 2.
            **kwargs: Additional keyword arguments passed to yaml.dump().

        Returns:
            The YAML string if path is None, otherwise None (file is written).
        """
        yaml_str = yaml.dump(self.to_dict(), indent=indent, **kwargs)
        if path is None:
            return yaml_str
        with open(path, "w") as f:
            f.write(yaml_str)

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2, **kwargs) -> str | None:
        """Convert the configuration to a JSON string or file.

        Args:
            path: Optional file path to write the JSON to. If None, returns the
                JSON string instead of writing to file.
            indent: Number of spaces for JSON indentation. Defaults to 2.
            **kwargs: Additional keyword arguments passed to json.dumps().

        Returns:
            The JSON string if path is None, otherwise None (file is written).
        """
        json_str = serialize_data(self.to_dict(), indent=indent, **kwargs)
        if path is None:
            return json_str
        with open(path, "w") as f:
            f.write(json_str)
