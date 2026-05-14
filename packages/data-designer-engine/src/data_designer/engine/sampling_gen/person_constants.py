# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple


class FakerPersonData(NamedTuple):
    sex: list[str] = ["Male", "Female"]

    us_locale_only: list[str] = [
        "state",
        "county",
        "unit",
        "middle_name",
        "ethnic_background",
        "ssn",
    ]

    location: list[str] = ["city", "state", "postcode"]

    bachelors: list[str] = [
        "stem",
        "business",
        "education",
        "arts_humanities",
        "stem_related",
    ]

    education_level: list[str] = [
        "secondary_education",
        "some_college",
        "bachelors",
        "associates",
        "graduate",
        "doctorate",
    ]

    marital_status: list[str] = [
        "married_present",
        "divorced",
        "never_married",
        "separated",
        "widowed",
    ]

    college_level: list[str] = ["bachelors", "graduate", "doctorate"]

    attr_map: dict[str, str] = {
        "street_number": "building_number",
        "occupation": "job",
    }


faker_constants = FakerPersonData()
