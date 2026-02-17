# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import random
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

import data_designer.lazy_heavy_imports as lazy


@functools.lru_cache(maxsize=1)
def _load_zip_area_code_data() -> tuple[dict[str, str], dict[str, int]]:
    """Load and cache the ZIP-to-area-code mapping from the parquet asset.

    Returns:
        A tuple of (area_code_map, population_map).
    """
    data = lazy.pd.read_parquet(Path(__file__).parent / "assets" / "zip_area_code_map.parquet")
    area_code_map = dict(zip(data["zipcode"], data["area_code"]))
    population_map = dict(zip(data["zipcode"], data["count"]))
    return area_code_map, population_map


def get_area_code(zip_prefix: str | None = None) -> str:
    """
    Sample an area code for the given ZIP code prefix, population-weighted.

    Args:
        zip_prefix: The prefix of a ZIP code, 5 digits or fewer. If None, sample from all ZIP codes.

    Returns:
        A sampled area code matching the prefix, population-weighted.
    """
    area_code_map, population_map = _load_zip_area_code_data()

    if zip_prefix is None:
        zipcodes, weights = zip(*population_map.items())
        zipcode = random.choices(zipcodes, weights=weights, k=1)[0]
        return str(area_code_map[zipcode])
    if len(zip_prefix) == 5:
        try:
            return str(area_code_map[zip_prefix])
        except KeyError:
            raise ValueError(f"ZIP code {zip_prefix} not found.")
    matching_zipcodes = [[z, c] for z, c in population_map.items() if z.startswith(zip_prefix)]
    zipcodes, weights = zip(*matching_zipcodes)
    if not zipcodes:
        raise ValueError(f"No ZIP codes found with prefix {zip_prefix}.")
    zipcode = random.choices(zipcodes, weights=weights, k=1)[0]
    return str(area_code_map[zipcode])


class PhoneNumber(BaseModel):
    """
    A phone number object that supports various formatting styles
    """

    country_code: str = Field(default="1")
    area_code: str
    prefix: str  # First part of the local number
    line_number: str  # Second part of the local number

    @field_validator("country_code", "area_code", "prefix", "line_number")
    @classmethod
    def validate_digits(cls, v):
        if not v.isdigit():
            raise ValueError("Must contain only digits")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code_length(cls, v):
        max_length = 3
        if len(v) > max_length:
            raise ValueError(f"Country code {v} is longer than {max_length} digits")
        return v

    def format(self, style: str = "dash") -> str:
        """
        Format the phone number according to the specified style.

        Args:
            style: One of "dash", "parentheses", "dot", "space", "no_separation",
                  "international_plus", "international"

        Returns:
            Formatted phone number string
        """
        if style == "dash":
            formatted = f"{self.area_code}-{self.prefix}-{self.line_number}"
        elif style == "parentheses":
            formatted = f"({self.area_code}) {self.prefix}-{self.line_number}"
        elif style == "dot":
            formatted = f"{self.area_code}.{self.prefix}.{self.line_number}"
        elif style == "space":
            formatted = f"{self.area_code} {self.prefix} {self.line_number}"
        elif style == "no_separation":
            formatted = f"{self.area_code}{self.prefix}{self.line_number}"
        elif style == "international_plus":
            cc = self.country_code or "1"  # Default to US/Canada
            formatted = f"+{cc} {self.area_code} {self.prefix} {self.line_number}"
        elif style == "international":
            cc = int(self.country_code or 1)  # Default to US/Canada
            formatted = f"{cc:03d} {self.area_code} {self.prefix} {self.line_number}"
        else:
            raise ValueError(f"Unsupported format style: {style}")

        return formatted

    @classmethod
    def from_area_code(cls, area_code: str) -> "PhoneNumber":
        prefix = str(random.randint(200, 1000))
        line_number = str(random.randint(0, 10000)).zfill(4)
        return PhoneNumber(area_code=area_code, prefix=prefix, line_number=line_number)

    @classmethod
    def from_zip_prefix(cls, zip_prefix: str) -> "PhoneNumber":
        """Create a phone number from the given ZIP code prefix."""
        area_code = get_area_code(zip_prefix)
        return cls.from_area_code(area_code)

    @classmethod
    def generate(cls) -> "PhoneNumber":
        """Create a random valid US phone number."""
        area_code = get_area_code()
        return cls.from_area_code(area_code)

    def __str__(self) -> str:
        return self.format("dash")

    def __repr__(self) -> str:
        return f"PhoneNumber({str(self)})"
