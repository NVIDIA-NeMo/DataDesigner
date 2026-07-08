# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pythonjsonlogger.json import JsonFormatter


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


class DataDesignerManagedHandler:
    """Marker mixin for handlers installed by ``configure_logging()``."""


class DataDesignerStreamHandler(DataDesignerManagedHandler, logging.StreamHandler):
    """Stream handler managed by Data Designer."""


class DataDesignerFileHandler(DataDesignerManagedHandler, logging.FileHandler):
    """File handler managed by Data Designer."""


class RandomEmoji:
    """A generator for various themed emoji collections."""

    def __init__(self) -> None:
        self._progress_style = random.choice(_PROGRESS_STYLES)

    def progress(self, percent: float) -> str:
        """Get a progress emoji based on completion percentage (0-100)."""
        phase_idx = min(int(percent / 25), len(self._progress_style) - 1)
        return self._progress_style[phase_idx]

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


def configure_logging(config: LoggingConfig | None = None) -> None:
    config = config or LoggingConfig.default()

    root_logger = logging.getLogger()

    # Remove and close all handlers replaced by this configuration.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

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


def is_logging_configured() -> bool:
    """Return whether Data Designer configured the root logger.

    Detected by the presence of a Data Designer-managed handler on the root
    logger. Logging configured through stdlib APIs (e.g. ``logging.basicConfig()``)
    is not detected.
    """
    return any(isinstance(handler, DataDesignerManagedHandler) for handler in logging.getLogger().handlers)


def reset_logging() -> None:
    """Remove Data Designer's logging configuration from the current process.

    Detaches handlers installed by ``configure_logging()`` from the root logger
    and returns the root and ``data_designer`` loggers to their default levels.
    Handlers installed by other code are left untouched. This does not restore
    logging configuration that existed before ``configure_logging()`` was called.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, DataDesignerManagedHandler):
            root_logger.removeHandler(handler)
            handler.close()
    root_logger.setLevel(logging.WARNING)
    logging.getLogger("data_designer").setLevel(logging.NOTSET)


def quiet_noisy_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)


def _create_handler(output_config: OutputConfig) -> logging.Handler:
    if isinstance(output_config.destination, Path):
        handler = DataDesignerFileHandler(str(output_config.destination))
    else:
        handler = DataDesignerStreamHandler()

    if output_config.structured:
        formatter = _make_json_formatter()
    else:
        formatter = _make_stream_formatter()

    handler.setFormatter(formatter)
    return handler


def _make_json_formatter() -> logging.Formatter:
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    return JsonFormatter(log_format)


def _make_stream_formatter() -> logging.Formatter:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    time_format = "%H:%M:%S"
    return logging.Formatter(log_format, time_format)


_DEFAULT_NOISY_LOGGERS = ["httpx", "matplotlib", "mcp"]

LOG_INDENT = "  |-- "
LOG_DOUBLE_INDENT = "  |  |-- "

_PROGRESS_STYLES: list[list[str]] = [
    ["🌑", "🌘", "🌗", "🌖", "🌕"],  # Moon phases
    ["🌧️", "🌦️", "⛅", "🌤️", "☀️"],  # Weather (storm to sun)
    ["🥚", "🐣", "🐥", "🐤", "🐔"],  # Hatching (egg to chicken)
    ["🐱", "😺", "😸", "😼", "🦁"],  # Kitten to lion (sweet to fierce)
    ["😴", "🥱", "😐", "😊", "🤩"],  # Waking up (sleepy to excited)
    ["🚶", "🐴", "🚗", "✈️", "🚀"],  # Evolution of transportation
]
