# Person Sampling in Data Designer

Person sampling in Data Designer allows you to generate synthetic person data for your datasets. There are two distinct approaches, each with different capabilities and use cases.

## Overview

Data Designer provides two ways to generate synthetic people:

1. **Faker-based sampling** - Quick, basic PII generation for testing or when realistic demographic distributions are not relevant for your use case
2. **Nemotron Personas datasets** - Demographically accurate, rich persona data

---

## Approach 1: Faker-Based Sampling

### What It Does
Uses the Faker library to generate random personal information. The data is basic and not demographically accurate, but is useful for quick testing, prototyping, or when realistic demographic distributions are not relevant for your use case.

### Features
- Gives you access to person attributes that Faker exposes
- Quick to set up with no additional downloads
- Generates random names, emails, addresses, phone numbers, etc.
- Supports [all Faker-supported locales](https://faker.readthedocs.io/en/master/locales.html)
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
            locale="en_US",
            age_range=[25, 65],
            sex="Female",
        ),
    )
)
```

See the [`SamplerColumnConfig`](../api/columns.md#samplercolumnconfig) documentation for more details.

---

## Approach 2: Nemotron Personas Datasets

### What It Does
Uses curated Nemotron Personas datasets from NVIDIA GPU Cloud (NGC) to generate demographically accurate person data with rich personality profiles and behavioral characteristics.

The NGC datasets are extended versions of the [open-source Nemotron Personas datasets on HuggingFace](https://huggingface.co/collections/nvidia/nemotron-personas), with additional fields and enhanced data quality.

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

You need to download the Nemotron Personas datasets that you want to use from NGC, they are available [here](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=nemotron+personas)

1. **NGC API Key**: Obtain from [NVIDIA GPU Cloud](https://ngc.nvidia.com/)
2. **NGC CLI**: [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli)

### Setup Instructions

#### Step 1: Set Your NGC API Key
```bash
export NGC_API_KEY="your-ngc-api-key-here"
```

#### Step 2: Download Nemotron Personas Datasets
Use the NGC CLI to download the datasets:
```bash
# For Nemotron Personas USA
ngc registry resource download-version "nvidia/nemotron-personas/nemotron-personas-dataset-en_us"

# For Nemotron Personas IN
ngc registry resource download-version "nvidia/nemotron-personas/nemotron-personas-dataset-hi_deva_in"
ngc registry resource download-version "nvidia/nemotron-personas/nemotron-personas-dataset-hi_latn_in"
ngc registry resource download-version "nvidia/nemotron-personas/nemotron-personas-dataset-en_in"

# For Nemotron Personas JP
ngc registry resource download-version "nvidia/nemotron-personas/nemotron-personas-dataset-ja_jp"
```

Then move the downloaded dataset to the Data Designer managed assets directory:
```bash
mkdir -p ~/.data-designer/managed-assets/datasets/
mv nemotron-personas-dataset-*/*.parquet ~/.data-designer/managed-assets/datasets/
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
            locale="en_US",
            sex="Female",
            age_range=[25, 45],
            with_synthetic_personas=True,
        ),
    )
)
```

See the [`SamplerColumnConfig`](../api/columns.md#samplercolumnconfig) documentation for more details.

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
| `national_id` | string |

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

