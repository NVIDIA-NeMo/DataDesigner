# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
