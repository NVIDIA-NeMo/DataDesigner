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
    # Universal demographic fields (present in every managed locale)
    "uuid",
    "first_name",
    "middle_name",
    "last_name",
    "sex",
    "age",
    "marital_status",
    "postcode",
    "city",
    "district",
    "region",
    "country",
    "street_name",
    "street_number",
    "unit",
    "bachelors_field",
    "education_level",
    "occupation",
    # Runtime-generated / attached fields
    "locale",
    "birth_date",
    "state",
    "email_address",
    "phone_number",
    "national_id",
    # en_US + en_SG
    "ethnic_background",
    # en_SG-specific
    "industry",
    "preferred_english_name",
    # fr_FR-specific
    "first_name_heritage",
    "name_heritage",
    "is_first_gen_immigrant",
    "household_type",
    "monthly_income_eur",
    # ja_JP-specific
    "area",
    # ko_KR-specific
    "blood_pressure_status",
    "blood_sugar_status",
    "bmi_status",
    "drinking_status",
    "economic_activity_status",
    "family_type",
    "housing_tenure",
    "housing_type",
    "income_bracket",
    "military_status",
    "smoking_status",
    "waist_status",
    # pt_BR-specific
    "race",
    # India locales (en_IN, hi_Deva_IN, hi_Latn_IN)
    "education_degree",
    "first_language",
    "second_language",
    "third_language",
    "zone",
    # Shared across India locales, en_SG, and pt_BR
    "religion",
]

PERSONA_FIELDS = [
    # Universal persona fields
    "persona",
    "detailed_persona",
    "professional_persona",
    "career_goals_and_ambitions",
    "cultural_background",
    "arts_persona",
    "culinary_persona",
    "finance_persona",
    "healthcare_persona",
    "sports_persona",
    "travel_persona",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "skills_and_expertise",
    "skills_and_expertise_list",
    # Big Five personality traits
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    # ja_JP-specific
    "aspects",
    "digital_skill",
    # ko_KR-specific
    "family_persona",
    # Shared across India locales, en_SG, and pt_BR
    "religious_persona",
    "religious_background",
    # India locales only (en_IN, hi_Deva_IN, hi_Latn_IN)
    "linguistic_persona",
    "linguistic_background",
]
