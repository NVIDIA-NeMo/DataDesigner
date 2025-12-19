# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pythonjsonlogger import jsonlogger


@dataclass
class LoggerConfig:
    name: str
    level: str


@dataclass
class OutputConfig:
    destination: TextIO | Path
    structured: bool


@dataclass
class LoggingConfig:
    logger_configs: list[LoggerConfig]
    output_configs: list[OutputConfig]
    root_level: str = "INFO"
    to_silence: list[str] = field(default_factory=lambda: _DEFAULT_NOISY_LOGGERS)

    @classmethod
    def default(cls):
        return LoggingConfig(
            logger_configs=[LoggerConfig(name="data_designer", level="INFO")],
            output_configs=[OutputConfig(destination=sys.stderr, structured=False)],
        )

    @classmethod
    def debug(cls):
        return LoggingConfig(
            logger_configs=[LoggerConfig(name="data_designer", level="DEBUG")],
            output_configs=[OutputConfig(destination=sys.stderr, structured=False)],
        )


class RandomEmoji:
    """A generator for various themed emoji collections."""

    @staticmethod
    def cooking() -> str:
        """Get a random cooking or food preparation emoji."""
        return random.choice(["👨‍🍳", "👩‍🍳", "🍳", "🥘", "🍲", "🔪", "🥄", "🍴", "⏲️", "🥗"])

    @staticmethod
    def data() -> str:
        """Get a random data or analytics emoji."""
        return random.choice(["📊", "📈", "📉", "💾", "💿", "📀", "🗄️", "📁", "📂", "🗃️"])

    @staticmethod
    def generating() -> str:
        """Get a random generating or creating emoji."""
        return random.choice(["🏭", "⚙️", "🔨", "🛠️", "🏗️", "🎨", "✍️", "📝", "🔧", "⚒️"])

    @staticmethod
    def loading() -> str:
        """Get a random loading or waiting emoji."""
        return random.choice(["⏳", "⌛", "🔄", "♻️", "🔃", "⏰", "⏱️", "⏲️", "📡", "🌀"])

    @staticmethod
    def magic() -> str:
        """Get a random magical or special effect emoji."""
        return random.choice(["✨", "⭐", "🌟", "💫", "🪄", "🔮", "🎩", "🌈", "💎", "🦄"])

    @staticmethod
    def previewing() -> str:
        """Get a random previewing or looking ahead emoji."""
        return random.choice(["👀", "📺", "🔁", "👁️", "🔭", "🕵️", "🧐", "📸", "🎥", "🖼️"])

    @staticmethod
    def speed() -> str:
        """Get a random speed or fast emoji."""
        return random.choice(["⚡", "💨", "🏃", "🏎️", "🚄", "✈️", "💥", "⏩", "🏃‍♂️", "🏃‍♀️"])

    @staticmethod
    def start() -> str:
        """Get a random emoji representing starting or launching something."""
        return random.choice(["🚀", "▶️", "🎬", "🌅", "🏁", "🎯", "🚦", "🔔", "📣", "🎺"])

    @staticmethod
    def success() -> str:
        """Get a random success or celebration emoji."""
        return random.choice(["🎉", "🎊", "👏", "🙌", "🎆", "🍾", "☀️", "🏆", "✅", "🥳"])

    @staticmethod
    def thinking() -> str:
        """Get a random thinking or processing emoji."""
        return random.choice(["🤔", "💭", "🧠", "💡", "🔍", "🔎", "🤨", "🧐", "📝", "🧮"])

    @staticmethod
    def working() -> str:
        """Get a random working or in-progress emoji."""
        return random.choice(["⚙️", "🔧", "🔨", "⚒️", "🛠️", "💼", "👷", "🏗️", "🪛", "👨‍💻"])


def configure_logging(config: LoggingConfig) -> None:
    root_logger = logging.getLogger()

    # Remove all handlers
    root_logger.handlers.clear()

    # Create and attach handler(s)
    handlers = [_create_handler(output_config) for output_config in config.output_configs]
    for handler in handlers:
        root_logger.addHandler(handler)

    # Set levels
    root_logger.setLevel(config.root_level)
    for logger_config in config.logger_configs:
        logger = logging.getLogger(logger_config.name)
        logger.setLevel(logger_config.level)

    # Adjust noisy loggers
    for name in config.to_silence:
        quiet_noisy_logger(name)


def quiet_noisy_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)


def _create_handler(output_config: OutputConfig) -> logging.Handler:
    if isinstance(output_config.destination, Path):
        handler = logging.FileHandler(str(output_config.destination))
    else:
        handler = logging.StreamHandler()

    if output_config.structured:
        formatter = _make_json_formatter()
    else:
        formatter = _make_stream_formatter()

    handler.setFormatter(formatter)
    return handler


def _make_json_formatter() -> logging.Formatter:
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    return jsonlogger.JsonFormatter(log_format)


def _make_stream_formatter() -> logging.Formatter:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    time_format = "%H:%M:%S"
    return logging.Formatter(log_format, time_format)


_DEFAULT_NOISY_LOGGERS = ["httpx", "matplotlib"]
