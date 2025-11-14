# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

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
    def value(self, val: T) -> None:
        """Set and validate the field value."""
        if self.validator:
            # For string validators, convert to string first if needed
            val_str = str(val) if not isinstance(val, str) else val
            is_valid, error_msg = self.validator(val_str)
            if not is_valid:
                raise ValidationError(error_msg or "Invalid value")
        self._value = val

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
        from data_designer.cli.ui import BACK, prompt_text_input

        result = prompt_text_input(
            self.prompt,
            default=self.default,
            validator=self.validator,
            mask=self.mask,
            completions=self.completions,
            allow_back=allow_back,
        )

        if result is BACK:
            return BACK

        return result


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
        from data_designer.cli.ui import BACK, select_with_arrows

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
        from data_designer.cli.ui import BACK, prompt_text_input

        default_str = str(self.default) if self.default is not None else None

        result = prompt_text_input(
            self.prompt,
            default=default_str,
            validator=self.validator,
            allow_back=allow_back,
        )

        if result is BACK:
            return BACK

        return float(result) if result else None
