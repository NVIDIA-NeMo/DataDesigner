# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from data_designer.cli.forms.form import Form
from data_designer.cli.ui import confirm_action, print_error

T = TypeVar("T", bound=BaseModel)


class FormBuilder(ABC, Generic[T]):
    """Abstract base for building interactive configuration forms."""

    def __init__(self, title: str):
        self.title = title

    @abstractmethod
    def create_form(self, initial_data: dict[str, Any] | None = None) -> Form:
        """Create the form for this configuration."""

    @abstractmethod
    def build_config(self, form_data: dict[str, Any]) -> T:
        """Build a configuration object from form data."""

    def run(self, initial_data: dict[str, Any] | None = None) -> T | None:
        """Run the interactive form and return configured object."""
        form = self.create_form(initial_data)

        # Pre-populate form with initial data
        if initial_data:
            form.set_values(initial_data)

        while True:
            result = form.prompt_all(allow_back=True)

            if result is None:
                if confirm_action("Cancel configuration?", default=False):
                    return None
                continue

            try:
                config = self.build_config(result)
                return config
            except Exception as e:
                print_error(f"Configuration error: {e}")
                if not confirm_action("Try again?", default=True):
                    return None
