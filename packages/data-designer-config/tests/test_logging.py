# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import tempfile
from pathlib import Path

import pytest

from data_designer.logging import (
    LoggerConfig,
    LoggingConfig,
    OutputConfig,
    RandomEmoji,
    configure_logging,
    quiet_noisy_logger,
)


@pytest.fixture
def stub_default_logging_config():
    """Fixture for default logging configuration."""
    return LoggingConfig(
        logger_configs=[LoggerConfig(name="data_designer", level="INFO")],
        output_configs=[OutputConfig(destination=sys.stderr, structured=False)],
        root_level="INFO",
    )


@pytest.fixture
def stub_debug_logging_config():
    """Fixture for debug logging configuration."""
    return LoggingConfig(
        logger_configs=[LoggerConfig(name="data_designer", level="DEBUG")],
        output_configs=[OutputConfig(destination=sys.stderr, structured=False)],
        root_level="DEBUG",
    )


def test_logger_config():
    config = LoggerConfig(name="test_logger", level="DEBUG")
    assert config.name == "test_logger"
    assert config.level == "DEBUG"


def test_output_config_stream():
    config = OutputConfig(destination=sys.stderr, structured=False)
    assert config.destination == sys.stderr
    assert config.structured is False


def test_output_config_file():
    path = Path("/tmp/test.log")
    config = OutputConfig(destination=path, structured=True)
    assert config.destination == path
    assert config.structured is True


def test_logging_config_default(stub_default_logging_config):
    config = stub_default_logging_config
    assert len(config.logger_configs) == 1
    assert config.logger_configs[0].name == "data_designer"
    assert config.logger_configs[0].level == "INFO"
    assert len(config.output_configs) == 1
    assert config.output_configs[0].destination == sys.stderr
    assert config.output_configs[0].structured is False
    assert config.root_level == "INFO"


def test_logging_config_debug(stub_debug_logging_config):
    config = stub_debug_logging_config
    assert len(config.logger_configs) == 1
    assert config.logger_configs[0].name == "data_designer"
    assert config.logger_configs[0].level == "DEBUG"
    assert len(config.output_configs) == 1
    assert config.output_configs[0].destination == sys.stderr
    assert config.output_configs[0].structured is False


def test_logging_config_custom():
    config = LoggingConfig(
        logger_configs=[
            LoggerConfig(name="logger1", level="INFO"),
            LoggerConfig(name="logger2", level="DEBUG"),
        ],
        output_configs=[OutputConfig(destination=sys.stdout, structured=True)],
        root_level="WARNING",
        to_silence=["noisy_logger"],
    )
    assert len(config.logger_configs) == 2
    assert config.root_level == "WARNING"
    assert config.to_silence == ["noisy_logger"]


def test_configure_logging_basic(stub_default_logging_config):
    config = stub_default_logging_config
    configure_logging(config)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO
    assert len(root_logger.handlers) >= 1

    ndd_logger = logging.getLogger("data_designer")
    assert ndd_logger.level == logging.INFO


def test_configure_logging_with_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        config = LoggingConfig(
            logger_configs=[LoggerConfig(name="test", level="DEBUG")],
            output_configs=[OutputConfig(destination=tmp_path, structured=False)],
            root_level="DEBUG",
        )
        configure_logging(config)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        assert any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_configure_logging_structured():
    config = LoggingConfig(
        logger_configs=[LoggerConfig(name="test", level="INFO")],
        output_configs=[OutputConfig(destination=sys.stderr, structured=True)],
    )
    configure_logging(config)

    root_logger = logging.getLogger()
    assert len(root_logger.handlers) >= 1
    # Check that the formatter is a JSON formatter
    handler = root_logger.handlers[0]
    assert handler.formatter is not None


def test_configure_logging_multiple_handlers():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        config = LoggingConfig(
            logger_configs=[LoggerConfig(name="test", level="INFO")],
            output_configs=[
                OutputConfig(destination=sys.stderr, structured=False),
                OutputConfig(destination=tmp_path, structured=True),
            ],
        )
        configure_logging(config)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) >= 2
    finally:
        tmp_path.unlink(missing_ok=True)


def test_quiet_noisy_logger():
    logger_name = "test_noisy_logger_unique"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    quiet_noisy_logger(logger_name)

    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 0


def test_configure_logging_silences_default_noisy_loggers(stub_default_logging_config):
    config = stub_default_logging_config
    configure_logging(config)

    httpx_logger = logging.getLogger("httpx")
    matplotlib_logger = logging.getLogger("matplotlib")

    assert httpx_logger.level == logging.WARNING
    assert matplotlib_logger.level == logging.WARNING


@pytest.mark.parametrize(
    "emoji_method",
    [
        RandomEmoji.cooking,
        RandomEmoji.data,
        RandomEmoji.generating,
        RandomEmoji.loading,
        RandomEmoji.magic,
        RandomEmoji.previewing,
        RandomEmoji.speed,
        RandomEmoji.start,
        RandomEmoji.success,
        RandomEmoji.thinking,
        RandomEmoji.working,
    ],
)
def test_random_emoji_methods(emoji_method):
    emoji = emoji_method()
    assert emoji is not None
    assert len(emoji) > 0


def test_random_emoji_randomness():
    # Test that emoji methods actually return different values (with high probability)
    emojis = [RandomEmoji.magic() for _ in range(100)]
    # If we get 100 samples, we should get at least 2 different emojis
    assert len(set(emojis)) > 1


def test_random_emoji_progress_returns_valid_emoji() -> None:
    emoji_gen = RandomEmoji()
    emoji = emoji_gen.progress(50.0)
    assert emoji is not None
    assert len(emoji) > 0


def test_random_emoji_progress_is_deterministic() -> None:
    emoji_gen = RandomEmoji()
    # Same percentage should always return the same emoji for a given instance
    assert emoji_gen.progress(0.0) == emoji_gen.progress(0.0)
    assert emoji_gen.progress(50.0) == emoji_gen.progress(50.0)
    assert emoji_gen.progress(100.0) == emoji_gen.progress(100.0)


def test_random_emoji_progress_phases_are_distinct() -> None:
    emoji_gen = RandomEmoji()
    # Each 25% phase should return a different emoji
    phase_emojis = [
        emoji_gen.progress(0.0),  # phase 0
        emoji_gen.progress(25.0),  # phase 1
        emoji_gen.progress(50.0),  # phase 2
        emoji_gen.progress(75.0),  # phase 3
        emoji_gen.progress(100.0),  # phase 4
    ]
    # All 5 phases should have distinct emojis
    assert len(set(phase_emojis)) == 5


def test_random_emoji_progress_phase_boundaries() -> None:
    emoji_gen = RandomEmoji()
    # Values within the same phase should return the same emoji
    assert emoji_gen.progress(0.0) == emoji_gen.progress(24.9)
    assert emoji_gen.progress(25.0) == emoji_gen.progress(49.9)
    assert emoji_gen.progress(50.0) == emoji_gen.progress(74.9)
    assert emoji_gen.progress(75.0) == emoji_gen.progress(99.9)
    # Phase transitions should return different emojis
    assert emoji_gen.progress(24.9) != emoji_gen.progress(25.0)
    assert emoji_gen.progress(49.9) != emoji_gen.progress(50.0)
    assert emoji_gen.progress(74.9) != emoji_gen.progress(75.0)
    assert emoji_gen.progress(99.9) != emoji_gen.progress(100.0)


def test_random_emoji_progress_clamps_over_100() -> None:
    emoji_gen = RandomEmoji()
    emoji_100 = emoji_gen.progress(100.0)
    emoji_over = emoji_gen.progress(150.0)
    # Both should return the same final emoji
    assert emoji_100 == emoji_over
