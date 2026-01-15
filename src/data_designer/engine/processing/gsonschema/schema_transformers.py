# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

from data_designer.engine.processing.gsonschema.types import JSONSchemaT


def _is_bare_dictionary_schema(schema_part: Any) -> bool:
    """Classify bare dictionary schemas

    A bare dictionary schema is one which looks like the following:

        { "title": ... , "type": "object" }
        { "type": "object" }

    These schemas do not specify any "properties", just their existence.
    """
    if not isinstance(schema_part, dict):
        return False

    if schema_part.get("type") != "object":
        return False

    if ("title" in schema_part and len(schema_part) == 2) or (len(schema_part) == 1):
        return True

    return False


def forbid_additional_properties(schema: JSONSchemaT) -> JSONSchemaT:
    """Transform the provided schema into one which forbids additional properties.

    Args:
        schema (JSONSchemaT): A JSONSchema to transform.

    Returns:
        JSONSchemaT: A new JSONSchema matching the provided one, but
            with `additionalProperties: False` set everywhere.
    """
    new_schema = deepcopy(schema)

    def _enforce(schema_part: Any) -> None:
        if isinstance(schema_part, dict):
            if schema_part.get("type") == "object" or "properties" in schema_part:
                ## We need to handle the special case that the schema specifies just a bare
                ## dictionary. In those cases, the implication is that _all_ dictionaries
                ## are valid, so we should not forbid extra properties in that case.
                allow_additional_properties = _is_bare_dictionary_schema(schema_part)
                schema_part["additionalProperties"] = allow_additional_properties

            # Traverse into nested schemas.
            for key, value in schema_part.items():
                if key in ("properties", "patternProperties"):
                    if isinstance(value, dict):
                        for sub_schema in value.values():
                            _enforce(sub_schema)
                elif key == "items":
                    if isinstance(value, dict):
                        _enforce(value)
                    elif isinstance(value, list):
                        for item in value:
                            _enforce(item)
                elif key in ("allOf", "anyOf", "oneOf"):
                    if isinstance(value, list):
                        for item in value:
                            _enforce(item)
                elif key in ("not", "if", "then", "else"):
                    if isinstance(value, dict):
                        _enforce(value)
                elif key == "$defs":
                    for sub_schema in value.values():
                        _enforce(sub_schema)

        elif isinstance(schema_part, list):
            for item in schema_part:
                _enforce(item)

    _enforce(new_schema)
    return new_schema
