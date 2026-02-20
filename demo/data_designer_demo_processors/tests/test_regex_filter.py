from unittest.mock import MagicMock

import pandas as pd
import pytest
from data_designer_demo_processors.regex_filter.config import RegexFilterProcessorConfig
from data_designer_demo_processors.regex_filter.impl import RegexFilterProcessor


@pytest.fixture()
def sample_data():
    return pd.DataFrame({"text": ["hello world", "foo bar", "hello again", "baz"]})


@pytest.fixture()
def resource_provider():
    return MagicMock()


class TestRegexFilterConfig:
    def test_defaults(self):
        cfg = RegexFilterProcessorConfig(name="f", column="text", pattern="hello")
        assert cfg.processor_type == "regex-filter"
        assert cfg.invert is False

    def test_invert(self):
        cfg = RegexFilterProcessorConfig(name="f", column="text", pattern="hello", invert=True)
        assert cfg.invert is True


class TestRegexFilterProcessor:
    def test_filter_matching(self, sample_data, resource_provider):
        cfg = RegexFilterProcessorConfig(name="f", column="text", pattern="hello")
        proc = RegexFilterProcessor(cfg, resource_provider)
        result = proc.process_before_batch(sample_data)
        assert list(result["text"]) == ["hello world", "hello again"]

    def test_filter_inverted(self, sample_data, resource_provider):
        cfg = RegexFilterProcessorConfig(name="f", column="text", pattern="hello", invert=True)
        proc = RegexFilterProcessor(cfg, resource_provider)
        result = proc.process_before_batch(sample_data)
        assert list(result["text"]) == ["foo bar", "baz"]

    def test_filter_no_match(self, sample_data, resource_provider):
        cfg = RegexFilterProcessorConfig(name="f", column="text", pattern="xyz")
        proc = RegexFilterProcessor(cfg, resource_provider)
        result = proc.process_before_batch(sample_data)
        assert len(result) == 0

    def test_regex_pattern(self, sample_data, resource_provider):
        cfg = RegexFilterProcessorConfig(name="f", column="text", pattern="^hello")
        proc = RegexFilterProcessor(cfg, resource_provider)
        result = proc.process_before_batch(sample_data)
        assert list(result["text"]) == ["hello world", "hello again"]
