# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This file contains all possible fields that:

    1. Exist in a managed PII + persona dataset
    2. Are included in the final generated dataset

Do not add any other code or logic in this file.
"""

from __future__ import annotations

REQUIRED_FIELDS = {"first_name", "last_name", "age", "locale"}

PII_FIELDS = [
    # Core demographic fields
    "uuid",
    "first_name",
    "middle_name",
    "last_name",
    "sex",
    "age",
    "birth_date",
    "marital_status",
    "postcode",
    "city",
    "region",
    "country",
    "locale",
    "bachelors_field",
    "education_level",
    "occupation",
    "national_id",
    # US-specific fields
    "street_name",
    "street_number",
    "unit",
    "state",
    "email_address",
    "phone_number",
    # Brazil-specific fields
    "race",
    # Japan-specific fields
    "area",
    "prefecture",
    "zone",
    # Brazil and India shared fields
    "religion",
    # India-specific fields
    "district",
    "education_degree",
    "first_language",
    "second_language",
    "third_language",
]

PERSONA_FIELDS = [
    # Core persona fields
    "persona",
    "career_goals_and_ambitions",
    "arts_persona",
    "culinary_persona",
    "cultural_background",
    "detailed_persona",
    "finance_persona",
    "healthcare_persona",
    "hobbies_and_interests_list",
    "hobbies_and_interests",
    "professional_persona",
    "skills_and_expertise_list",
    "skills_and_expertise",
    "sports_persona",
    "travel_persona",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    # Japan-specific persona fields
    "aspects",
    "digital_skills",
    # Brazil and India shared persona fields
    "religious_persona",
    "religious_background",
    # India-specific persona fields
    "linguistic_persona",
    "linguistic_background",
]
