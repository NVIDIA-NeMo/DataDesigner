# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This file contains all possible fields that:

    1. Exist in a managed PII + persona dataset
    2. Are included in the final generated dataset

Do not add any other code or logic in this file.
"""

REQUIRED_FIELDS = {"first_name", "last_name", "age", "locale"}


PII_FIELDS = [
    "uuid",
    "first_name",
    "middle_name",
    "last_name",
    "sex",
    "age",
    "birth_date",
    "marital_status",
    "street_name",
    "street_number",
    "unit",
    "postcode",
    "region",
    "city",
    "district",
    "country",
    "area",
    "zone",
    "bachelors_field",
    "education_degree",
    "education_level",
    "occupation",
    "locale",
]


PERSONA_FIELDS = [
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
]
