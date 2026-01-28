# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from data_designer.engine.processing.ginja.exceptions import RecordContentsError
from data_designer.engine.processing.ginja.record import sanitize_record


def test_sanitize_record():
    @dataclass
    class Bar:
        foo: int
        you: list[int]
        unicode_str: str

    record = Bar(foo=1, you=[1, 2, 3], unicode_str="你好")
    valid_input = {"foo": 1, "you": [1, 2, 3], "unicode_str": "你好"}
    assert sanitize_record(valid_input) == valid_input

    invalid_input = {"test": "case", "BarClass": record}
    with pytest.raises(RecordContentsError):
        sanitize_record(invalid_input)
