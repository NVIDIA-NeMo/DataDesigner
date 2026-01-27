# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from data_designer.config.utils.io_helpers import serialize_data
from data_designer.engine.processing.ginja.exceptions import RecordContentsError


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
