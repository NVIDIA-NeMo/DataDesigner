# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

from data_designer.config.utils.io_helpers import serialize_data
from data_designer.engine.processing.ginja.exceptions import RecordContentsError


class TemplateValue:
    """Wraps a value so Jinja2 can drill into nested dicts via dot notation.

    Jinja2 resolves {{ x.y }} through __getattr__, so wrapping a dict in
    TemplateValue lets {{ record.quality.score }} traverse the nested
    structure. Each lookup returns a new TemplateValue, propagating the
    behavior down. When Jinja2 interpolates the final value (calls __str__),
    str_fn controls the conversion - allowing callers to apply custom
    escaping (e.g. JSON escaping for schema transform templates).
    """

    __slots__ = ("_value", "_str_fn")

    def __init__(self, value: Any, str_fn: Callable[[Any], str] = str) -> None:
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_str_fn", str_fn)

    def __getattr__(self, name: str) -> TemplateValue:
        if isinstance(self._value, dict) and name in self._value:
            return TemplateValue(self._value[name], str_fn=self._str_fn)
        raise AttributeError(f"'{type(self._value).__name__} object' has no attribute '{name}'")

    def __getitem__(self, key: Any) -> TemplateValue:
        return TemplateValue(self._value[key], str_fn=self._str_fn)

    def __str__(self) -> str:
        return self._str_fn(self._value)

    def __iter__(self) -> Iterator[TemplateValue]:
        for item in self._value:
            yield TemplateValue(item, str_fn=self._str_fn)

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


def wrap_record(record: dict[str, Any], str_fn: Callable[[Any], str] = str) -> dict[str, TemplateValue]:
    """Wrap all values in a record as TemplateValues for nested Jinja2 access."""
    return {k: TemplateValue(v, str_fn=str_fn) for k, v in record.items()}


def sanitize_record(record: dict) -> dict:
    """Sanitize a record into basic types.

    To prevent any unexpected attributes from being callable from
    the template, we apply a serdes step to ensure that the record
    used as context for the rendering step consists of basic
    python types (e.g. those that can be represented via JSON).

    Args:
        record (dict): A dictionary object which can be serialized.

    Raises:
        RecordContentsError if the record contents are not able
            to be represented with JSON.
    """
    try:
        ser = serialize_data(record)
    except (TypeError, ValueError) as e:
        raise RecordContentsError("Unexpected unserializable content found in record.") from e

    return json.loads(ser)
