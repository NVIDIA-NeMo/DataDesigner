# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from data_designer.cli.ui import BACK, prompt_text_input, select_with_arrows
from data_designer.cli.utils import validate_numeric_range

T = TypeVar("T")


class ValidationError(Exception):
    """Field validation error."""


class Field(ABC, Generic[T]):
    """Base class for form fields."""

    def __init__(
        self,
        name: str,
        prompt: str,
        default: T | None = None,
        required: bool = True,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        help_text: str | None = None,
    ):
        self.name = name
        self.prompt = prompt
        self.default = default
        self.required = required
        self.validator = validator
        self.help_text = help_text
        self._value: T | None = None

    @property
    def value(self) -> T | None:
        """Get the current field value."""
        return self._value

    @value.setter
    def value(self, val: T | str) -> None:
        """Set and validate the field value. Converts empty strings to None for optional fields."""
        # Handle empty string for optional fields (clearing the value)
        if val == "" and not self.required:
            self._value = None
            return

        # Standard validation for non-empty values
        if self.validator:
            # For string validators, convert to string first if needed
            val_str = str(val) if not isinstance(val, str) else val
            is_valid, error_msg = self.validator(val_str)
            if not is_valid:
                raise ValidationError(error_msg or "Invalid value")
        self._value = val

    def _build_prompt_text(self) -> str:
        """Build prompt text with current value information."""
        has_current_value = self.default is not None

        if has_current_value:
            # Show as "current" instead of "default" with dimmed styling
            if not self.required:
                return f"{self.prompt} <dim>(current value: {self.default}, type 'clear' to remove)</dim>"
            return f"{self.prompt} <dim>(current value: {self.default})</dim>"

        return self.prompt

    def _handle_prompt_result(self, result: str | None | Any) -> str | None | Any:
        """Handle common prompt result logic (BACK, None, clear keywords, empty input)."""
        if result is BACK:
            return BACK

        if result is None:
            # User cancelled (ESC)
            return None

        # Check for special keywords to clear the value
        if result and result.lower() in ("clear", "none", "default"):
            return ""

        if not result:
            # Empty input: return current value if exists
            has_current_value = self.default is not None
            if has_current_value:
                return self.default
            return ""

        return result

    @abstractmethod
    def prompt_user(self, allow_back: bool = False) -> T | None | Any:
        """Prompt user for input."""


class TextField(Field[str]):
    """Text input field."""

    def __init__(
        self,
        name: str,
        prompt: str,
        default: str | None = None,
        required: bool = True,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        completions: list[str] | None = None,
        mask: bool = False,
        help_text: str | None = None,
    ):
        super().__init__(name, prompt, default, required, validator, help_text)
        self.completions = completions
        self.mask = mask

    def prompt_user(self, allow_back: bool = False) -> str | None | Any:
        """Prompt user for text input."""
        prompt_text = self._build_prompt_text()

        # Don't pass default to prompt_text_input to avoid duplicate "(default: X)" text
        result = prompt_text_input(
            prompt_text,
            default=None,
            validator=self.validator,
            mask=self.mask,
            completions=self.completions,
            allow_back=allow_back,
        )

        return self._handle_prompt_result(result)


class SelectField(Field[str]):
    """Selection field with arrow navigation."""

    def __init__(
        self,
        name: str,
        prompt: str,
        options: dict[str, str],
        default: str | None = None,
        required: bool = True,
        help_text: str | None = None,
    ):
        super().__init__(name, prompt, default, required, None, help_text)
        self.options = options

    def prompt_user(self, allow_back: bool = False) -> str | None | Any:
        """Prompt user for selection."""
        result = select_with_arrows(
            self.options,
            self.prompt,
            default_key=self.default,
            allow_back=allow_back,
        )

        if result is BACK:
            return BACK

        return result


class NumericField(Field[float]):
    """Numeric input field with range validation."""

    def __init__(
        self,
        name: str,
        prompt: str,
        default: float | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        required: bool = True,
        help_text: str | None = None,
    ):
        self.min_value = min_value
        self.max_value = max_value

        # Build validator based on range
        def range_validator(value: str) -> tuple[bool, str | None]:
            if not value and not required:
                return True, None
            # Allow special keywords to clear the value
            if value and value.lower() in ("clear", "none", "default"):
                return True, None
            if min_value is not None and max_value is not None:
                is_valid, parsed = validate_numeric_range(value, min_value, max_value)
                if not is_valid:
                    return False, f"Value must be between {min_value} and {max_value}"
                return True, None
            try:
                num = float(value)
                if min_value is not None and num < min_value:
                    return False, f"Value must be >= {min_value}"
                if max_value is not None and num > max_value:
                    return False, f"Value must be <= {max_value}"
                return True, None
            except ValueError:
                return False, "Must be a valid number"

        super().__init__(name, prompt, default, required, range_validator, help_text)

    def prompt_user(self, allow_back: bool = False) -> float | None | Any:
        """Prompt user for numeric input."""
        prompt_text = self._build_prompt_text()

        # Don't pass default to prompt_text_input to avoid duplicate "(default: X)" text
        result = prompt_text_input(
            prompt_text,
            default=None,
            validator=self.validator,
            allow_back=allow_back,
        )

        result = self._handle_prompt_result(result)

        # Return special values (BACK, None, empty string, defaults) as-is
        if result is BACK or result is None or result == "":
            return result

        # Convert numeric strings to float (but not if it's already a float from default)
        if isinstance(result, str):
            return float(result)

        return result
