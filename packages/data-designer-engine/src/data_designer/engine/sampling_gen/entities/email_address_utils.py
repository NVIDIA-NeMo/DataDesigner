# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import re
from datetime import date

import anyascii


def get_email_address(
    first_name: str,
    middle_name: str | None,
    last_name: str,
    age: int,
    birth_date: date,
) -> str:
    """
    Generate an email address based on a person's attributes.
    """

    domain = get_email_domain_by_age(age)
    username_base = get_email_basename_by_name(first_name, middle_name, last_name)
    suffix = get_email_suffix_by_birth_date(birth_date)

    # Combine to form email
    return f"{username_base}{suffix}@{domain}"


def get_email_domain_by_age(age: int) -> str:
    """
    Get a free email domain heuristically dependent on
    overall number of subscribers and user age.
    """

    # Common free email domains
    # Source: https://www.sellcell.com/blog/most-popular-email-provider-by-number-of-users/
    # Split heuristically into age demographics
    # Also adjusted to maintain the approximate 38/27/35 split between these groups
    email_domains_under_30 = {
        "gmail.com": 710,  # gmail.com total: 1500
        "icloud.com": 300,  # icloud.com total: 850
        "outlook.com": 50,  # outlook.com total: 200
        "hotmail.com": 40,  # hotmail.com total: 200
        "yahoo.com": 35,  # yahoo.com total: 230
        "protonmail.com": 20,  # protonmail.com total: 50
        "zoho.com": 3,  # zoho.com total: 15
        "gmx.com": 3,  # gmx.com total: 11
        "aol.com": 0.1,  # aol.com total: 1.5
    }
    email_domains_30_50 = {
        "gmail.com": 360,
        "icloud.com": 270,
        "outlook.com": 60,
        "hotmail.com": 50,
        "yahoo.com": 60,
        "protonmail.com": 18,
        "zoho.com": 7,
        "gmx.com": 4,
        "aol.com": 0.3,
    }
    email_domains_over_50 = {
        "gmail.com": 430,
        "icloud.com": 280,
        "outlook.com": 90,
        "hotmail.com": 110,
        "yahoo.com": 135,
        "protonmail.com": 12,
        "zoho.com": 5,
        "gmx.com": 4,
        "aol.com": 1.1,
    }

    if age < 30:
        return random.choices(
            list(email_domains_under_30.keys()),
            weights=list(email_domains_under_30.values()),
            k=1,
        )[0]
    elif age < 50:
        return random.choices(
            list(email_domains_30_50.keys()),
            weights=list(email_domains_30_50.values()),
            k=1,
        )[0]
    else:
        return random.choices(
            list(email_domains_over_50.keys()),
            weights=list(email_domains_over_50.values()),
            k=1,
        )[0]


def get_email_basename_by_name(first_name: str, middle_name: str | None, last_name: str) -> str:
    """
    Get a email address basename heuristically dependent on first and last name.

    Patterns include:
        - firstname.lastname
        - firstnamelastname
        - firstinitiallastname
        - firstname_lastname
        - lastnamefirstinitial
        - firstnamelastinitial
        - firstnamemiddlename
        - firstnamemiddleinitiallastname
        - firstnamemiddlenamelastname
    """
    # Normalize names (lowercase, remove spaces and special chars)
    first = re.sub(r"[^a-z0-9]", "", anyascii.anyascii(first_name).lower())
    last = re.sub(r"[^a-z0-9]", "", anyascii.anyascii(last_name).lower())
    assert len(first) > 0 and len(last) > 0, (
        "Both first and last name must be non-empty, after removing non-alphanumeric."
    )
    first_initial = first[0]
    last_initial = last[0]

    # Generate username patterns
    username_patterns = [
        f"{first}.{last}",
        f"{first}{last}",
        f"{first_initial}{last}",
        f"{first}_{last}",
        f"{last}{first_initial}",
        f"{first}{last_initial}",
    ]
    # Higher probability for more common patterns
    pattern_weights = [0.3, 0.2, 0.15, 0.1, 0.15, 0.1]
    if middle_name:
        middle = re.sub(r"[^a-z0-9]", "", anyascii.anyascii(middle_name).lower())
        middle_initial = middle[0]
        username_patterns.extend(
            [
                f"{first}{middle}",
                f"{first}{middle_initial}{last}",
                f"{first}{middle}{last}",
            ]
        )
        pattern_weights = [0.25, 0.17, 0.12, 0.08, 0.12, 0.08, 0.06, 0.06, 0.06]

    return random.choices(username_patterns, weights=pattern_weights, k=1)[0]


def get_email_suffix_by_birth_date(birth_date: date) -> str:
    """
    Get a email address suffix heuristically dependent on birth date.

    Suffices include:
        - Empty
        - Random 1-2 digit number
        - Last 2 digits of birth year
        - Full birth year
        - Birth day
    """
    # Suffix patterns (could be empty)
    birth_day = birth_date.day
    birth_year = birth_date.year
    birth_year_short = birth_year % 100
    suffix_patterns = [
        "",
        str(random.randint(1, 99)),
        f"{birth_year_short:02d}",
        str(birth_date.year),
        str(birth_day),
    ]
    suffix_weights = [0.4, 0.3, 0.1, 0.1, 0.1]

    # Select pattern and suffix based on weights
    return random.choices(suffix_patterns, weights=suffix_weights, k=1)[0]
