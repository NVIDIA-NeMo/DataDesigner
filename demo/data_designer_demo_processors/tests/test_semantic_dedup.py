from unittest.mock import MagicMock

import pandas as pd
import pytest
from data_designer_demo_processors.semantic_dedup.config import SemanticDedupProcessorConfig
from data_designer_demo_processors.semantic_dedup.impl import SemanticDedupProcessor


@pytest.fixture()
def resource_provider():
    return MagicMock()


class TestSemanticDedupConfig:
    def test_defaults(self):
        cfg = SemanticDedupProcessorConfig(name="d", column="text")
        assert cfg.processor_type == "semantic-dedup"
        assert cfg.similarity_threshold == 0.9
        assert cfg.model_name == "all-MiniLM-L6-v2"

    def test_custom_threshold(self):
        cfg = SemanticDedupProcessorConfig(name="d", column="text", similarity_threshold=0.8)
        assert cfg.similarity_threshold == 0.8


class TestSemanticDedupProcessor:
    def test_removes_near_duplicates(self, resource_provider):
        data = pd.DataFrame(
            {
                "text": [
                    "The cat sat on the mat",
                    "The cat was sitting on the mat",
                    "Python is a programming language",
                ]
            }
        )
        cfg = SemanticDedupProcessorConfig(name="d", column="text", similarity_threshold=0.85)
        proc = SemanticDedupProcessor(cfg, resource_provider)
        result = proc.process_after_generation(data)
        assert len(result) < len(data)
        assert "Python is a programming language" in result["text"].values

    def test_keeps_all_when_dissimilar(self, resource_provider):
        data = pd.DataFrame(
            {
                "text": [
                    "The weather is sunny today",
                    "Python is a programming language",
                    "Mount Everest is very tall",
                ]
            }
        )
        cfg = SemanticDedupProcessorConfig(name="d", column="text", similarity_threshold=0.95)
        proc = SemanticDedupProcessor(cfg, resource_provider)
        result = proc.process_after_generation(data)
        assert len(result) == 3

    def test_single_row(self, resource_provider):
        data = pd.DataFrame({"text": ["only one row"]})
        cfg = SemanticDedupProcessorConfig(name="d", column="text")
        proc = SemanticDedupProcessor(cfg, resource_provider)
        result = proc.process_after_generation(data)
        assert len(result) == 1
