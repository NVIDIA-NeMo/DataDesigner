# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from datetime import date

SSN_RANDOMIZATION_DATE = date(2011, 6, 25)

# Area number mapping by state code (pre-2011)
STATE_TO_AREA_SSN = {
    "NH": [1, 3],
    "ME": [4, 7],
    "VT": [8, 9],
    "MA": [10, 34],
    "RI": [35, 39],
    "CT": [40, 49],
    "NY": [50, 134],
    "NJ": [135, 158],
    "PA": [159, 211],
    "MD": [212, 220],
    "DE": [221, 222],
    "VA": [223, 231],
    "WV": [232, 236],
    "NC": [237, 246],
    "SC": [247, 251],
    "GA": [252, 260],
    "FL": [261, 267],
    "OH": [268, 302],
    "IN": [303, 317],
    "IL": [318, 361],
    "MI": [362, 386],
    "WI": [387, 399],
    "KY": [400, 407],
    "TN": [408, 415],
    "AL": [416, 424],
    "MS": [425, 428],
    "AR": [429, 432],
    "LA": [433, 439],
    "OK": [440, 448],
    "TX": [449, 467],
    "MN": [468, 477],
    "IA": [478, 485],
    "MO": [486, 500],
    "ND": [501, 502],
    "SD": [503, 504],
    "NE": [505, 508],
    "KS": [509, 515],
    "MT": [516, 517],
    "ID": [518, 519],
    "WY": [520, 520],
    "CO": [521, 524],
    "NM": [525, 527],
    "AZ": [526, 527],
    "UT": [528, 529],
    "NV": [530, 530],
    "WA": [531, 539],
    "OR": [540, 544],
    "CA": [545, 573],
    "AK": [574, 574],
    "HI": [575, 576],
    "DC": [577, 579],
    "VI": [580, 580],
    "PR": [580, 599],
    "GU": [586, 586],
    "AS": [586, 586],
}


def generate_ssn(state: str, birth_date: date) -> str:
    """
    Generate a synthetic SSN based on state and birth date.

    The first three digits are derived from the state the person lives in,
    if born after June 25, 2011, with an 80% chance. Otherwise, the first
    three digits are randomly chosen from the possible codes.

    Args:
        state (str): Two-letter state code (e.g., "NY", "CA")
        birth_date (date): Date of birth

    Returns:
        str: A formatted synthetic SSN in the format "XXX-XX-XXXX"

    """
    if birth_date < SSN_RANDOMIZATION_DATE:
        if random.random() < 0.3:
            # Maybe born in a different state
            area_range = random.choice(list(STATE_TO_AREA_SSN.values()))
        area_range = STATE_TO_AREA_SSN.get(state, [1, 899])
    else:
        area_range = [1, 899]
    area = 666
    while area == 666:
        # Unallowed area code
        area = random.randint(area_range[0], area_range[1])
    # Group number
    group = random.randint(1, 99)
    # Serial number
    serial = random.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"
