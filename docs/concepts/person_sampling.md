# Person Sampling in Data Designer

Person sampling in Data Designer allows you to generate synthetic person data for your datasets. There are two distinct approaches, each with different capabilities and use cases.

## Overview

Data Designer provides two ways to generate synthetic people:

1. **Faker-based sampling** - Quick, basic PII generation for testing
2. **Nemotron Personas datasets** - Demographically accurate, rich persona data

---

## Approach 1: Faker-Based Sampling

### What It Does
Uses the Faker library to generate random personal information. The data is basic and not demographically accurate, but is useful for quick testing and prototyping.

### Features
- Leverages all PII data features that Faker exposes
- Quick to set up with no additional downloads
- Generates random names, emails, addresses, phone numbers, etc.
- **Not demographically grounded** - data patterns don't reflect real-world demographics

### Usage Example
```python
from data_designer.essentials import (
    SamplerColumnConfig,
    SamplerType,
    PersonFromFakerSamplerParams,
)

config_builder.add_column(
    SamplerColumnConfig(
        name="customer",
        sampler_type=SamplerType.PERSON_FROM_FAKER,
        params=PersonFromFakerSamplerParams(
            locale="en_US",  # Any Faker-supported locale
            age_range=[25, 65],  # Optional: filter by age range
            sex="Female",  # Optional: filter by sex ("Male" or "Female")
        ),
    )
)
```

---

## Approach 2: Nemotron Personas Datasets

### What It Does
Uses curated Nemotron Personas datasets from NVIDIA GPU Cloud (NGC) to generate demographically accurate person data with rich personality profiles and behavioral characteristics.

### Features
- **Demographically accurate personal details**: Names, ages, sex, marital status, education, occupation based on census data
- **Rich persona details**: Comprehensive behavioral profiles including:
  - Big Five personality traits with scores
  - Cultural backgrounds and narratives
  - Skills and hobbies
  - Career goals and aspirations
  - Context-specific personas (professional, financial, healthcare, sports, arts, travel, culinary, etc.)
- Consistent, referenceable attributes across your dataset
- Grounded in real-world demographic distributions

### Prerequisites
1. **NGC API Key**: Obtain from [NVIDIA GPU Cloud](https://ngc.nvidia.com/)
2. **NGC CLI**: [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli)

### Setup Instructions

#### Step 1: Set Your NGC API Key
```bash
export NGC_API_KEY="your-ngc-api-key-here"
```

#### Step 2: Download Nemotron Personas Datasets
Use the Data Designer CLI to download the datasets:
```bash
ngc registry resource download-version "nvidia/nemo-microservices/nemotron-personas-dataset-en_us:0.0.6"
```

This will save the datasets to:
```
~/.data-designer/managed-assets/datasets/
```

#### Step 3: Use PersonSampler in Your Code
```python
from data_designer.essentials import (
    SamplerColumnConfig,
    SamplerType,
    PersonSamplerParams,
)

config_builder.add_column(
    SamplerColumnConfig(
        name="customer",
        sampler_type=SamplerType.PERSON,
        params=PersonSamplerParams(
            locale="en_US",  # Required: must be one of the managed dataset locales
            sex="Female",  # Optional: filter by sex ("Male" or "Female")
            age_range=[25, 45],  # Optional: filter by age range
            with_synthetic_personas=True,  # Optional: enable rich persona details
        ),
    )
)
```

### Available Data Fields

**Core Fields (all locales):**

| Field | Type | Notes |
|-------|------|-------|
| `uuid` | UUID | Unique identifier |
| `first_name` | string | |
| `middle_name` | string | |
| `last_name` | string | |
| `sex` | enum | "Male" or "Female" |
| `birth_date` | date | Derived: year, month, day |
| `street_number` | int | |
| `street_name` | string | |
| `unit` | string | Address line 2 |
| `city` | string | |
| `region` | string | Alias: state |
| `district` | string | Alias: county |
| `postcode` | string | Alias: zipcode |
| `country` | string | |
| `phone_number` | PhoneNumber | Derived: area_code, country_code, prefix, line_number |
| `marital_status` | string | Values: never_married, married_present, separated, widowed, divorced |
| `education_level` | string or None | |
| `bachelors_field` | string or None | |
| `occupation` | string or None | |
| `email_address` | string | |
| `national_id` | string | SSN for US locale |

**Japan-Specific Fields (`ja_JP`):**
- `area`

**India-Specific Fields (`en_IN`, `hi_IN`):**
- `religion` - Census-reported religion
- `education_degree` - Census-reported education degree
- `first_language` - Native language
- `second_language` - Second language (if applicable)
- `third_language` - Third language (if applicable)
- `zone` - Urban vs rural

**With Synthetic Personas Enabled:**
- Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) with t-scores and labels
- Cultural background narratives
- Skills and competencies
- Hobbies and interests
- Career goals
- Context-specific personas (professional, financial, healthcare, sports, arts & entertainment, travel, culinary, etc.)

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `locale` | str | Language/region code - must be one of: "en_US", "ja_JP", "en_IN", "hi_IN" |
| `sex` | str (optional) | Filter by "Male" or "Female" |
| `city` | str or list[str] (optional) | Filter by specific city or cities within locale |
| `age_range` | list[int] (optional) | Two-element list [min_age, max_age] (default: [18, 114]) |
| `with_synthetic_personas` | bool (optional) | Include rich personality profiles (default: False) |
| `select_field_values` | dict (optional) | Custom field-based filtering (e.g., {"state": ["NY", "CA"], "education_level": ["bachelors"]}) |
