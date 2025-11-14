# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from data_designer.cli.forms.field import Field
from data_designer.cli.ui import BACK, print_error


class Form:
    """A collection of fields forming a complete configuration form."""

    def __init__(self, name: str, fields: list[Field]):
        self.name = name
        self.fields = fields
        self._field_map = {f.name: f for f in fields}

    def get_field(self, name: str) -> Field | None:
        """Get a field by name."""
        return self._field_map.get(name)

    def get_values(self) -> dict[str, Any]:
        """Get all field values as a dictionary."""
        return {field.name: field.value for field in self.fields if field.value is not None}

    def set_values(self, values: dict[str, Any]) -> None:
        """Set field values from a dictionary."""
        for name, value in values.items():
            field = self.get_field(name)
            if field:
                field.value = value

    def prompt_all(self, allow_back: bool = True) -> dict[str, Any] | None:
        """Prompt user for all fields in sequence with back navigation."""
        field_index = 0

        while field_index < len(self.fields):
            field = self.fields[field_index]

            result = field.prompt_user(allow_back=allow_back and field_index > 0)

            if result is None:
                # User cancelled
                return None
            elif result is BACK:
                # Go back to previous field
                if field_index > 0:
                    field_index -= 1
                continue
            else:
                # Store value and move forward
                try:
                    field.value = result
                    field_index += 1
                except Exception as e:
                    print_error(f"Validation error: {e}")
                    continue

        return self.get_values()
