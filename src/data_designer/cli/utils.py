# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from data_designer.config.errors import InvalidConfigError, InvalidFileFormatError, InvalidFilePathError


def ensure_config_dir_exists(config_dir: Path) -> None:
    """Create configuration directory if it doesn't exist.

    Args:
        config_dir: Directory path to create
    """
    config_dir.mkdir(parents=True, exist_ok=True)


def load_config_file(file_path: Path) -> dict:
    """Load a YAML configuration file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        InvalidFilePathError: If file doesn't exist
        InvalidFileFormatError: If YAML is malformed
        InvalidConfigError: If file is empty
    """
    if not file_path.exists():
        raise InvalidFilePathError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path) as f:
            content = yaml.safe_load(f)

        if content is None:
            raise InvalidConfigError(f"Configuration file is empty: {file_path}")

        return content

    except yaml.YAMLError as e:
        raise InvalidFileFormatError(f"Invalid YAML format in {file_path}: {e}")


def save_config_file(file_path: Path, config: dict) -> None:
    """Save configuration to a YAML file.

    Args:
        file_path: Path where to save the file
        config: Configuration dictionary to save

    Raises:
        IOError: If file cannot be written
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            allow_unicode=True,
        )


def validate_url(url: str) -> bool:
    """Validate that a string is a valid URL.

    Args:
        url: URL string to validate

    Returns:
        True if valid URL, False otherwise
    """
    if not url:
        return False

    # Basic validation - must start with http:// or https://
    if not url.startswith(("http://", "https://")):
        return False

    # Must have at least a domain after the protocol
    parts = url.split("://", 1)
    if len(parts) != 2 or not parts[1]:
        return False

    return True


def validate_numeric_range(value: str, min_value: float, max_value: float) -> tuple[bool, float | None]:
    """Validate that a string is a valid number within a range.

    Args:
        value: String to validate and convert
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        Tuple of (is_valid, parsed_value)
        If invalid, parsed_value is None
    """
    try:
        num = float(value)
        if min_value <= num <= max_value:
            return True, num
        return False, None
    except ValueError:
        return False, None
