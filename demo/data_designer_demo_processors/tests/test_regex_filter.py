# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer_demo_processors.regex_filter import RegexFilterProcessor, RegexFilterProcessorConfig


@pytest.fixture
def stub_resource_provider():
    return Mock()


def test_regex_filter_keeps_matching_rows(stub_resource_provider):
    config = RegexFilterProcessorConfig(name="test", column="text", pattern=r"hello")
    processor = RegexFilterProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"text": ["hello world", "goodbye", "say hello", "nothing"]})
    result = processor.process(df)

    assert len(result) == 2
    assert list(result["text"]) == ["hello world", "say hello"]


def test_regex_filter_invert_keeps_non_matching(stub_resource_provider):
    config = RegexFilterProcessorConfig(name="test", column="text", pattern=r"hello", invert=True)
    processor = RegexFilterProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"text": ["hello world", "goodbye", "say hello", "nothing"]})
    result = processor.process(df)

    assert len(result) == 2
    assert list(result["text"]) == ["goodbye", "nothing"]


def test_regex_filter_missing_column_returns_unchanged(stub_resource_provider):
    config = RegexFilterProcessorConfig(name="test", column="missing", pattern=r"hello")
    processor = RegexFilterProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"text": ["hello world", "goodbye"]})
    result = processor.process(df)

    assert len(result) == 2


def test_regex_filter_complex_pattern(stub_resource_provider):
    config = RegexFilterProcessorConfig(name="test", column="email", pattern=r"^\w+@\w+\.\w+$")
    processor = RegexFilterProcessor(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"email": ["user@example.com", "invalid", "other@test.org", "bad@"]})
    result = processor.process(df)

    assert len(result) == 2
    assert list(result["email"]) == ["user@example.com", "other@test.org"]
