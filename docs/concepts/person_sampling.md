# Person Sampling in Data Designer

Person sampling in Data Designer allows you to generate synthetic person data for your datasets using the Faker library.

## Faker-Based Sampling

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
