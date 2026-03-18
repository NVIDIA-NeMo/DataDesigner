# Person Sampling Reference

## Sampler types

Prefer `"person"` when the locale is downloaded — it provides census-grounded demographics and optional personality traits. Fall back to `"person_from_faker"` when the locale isn't available.

| `sampler_type` | Params class | When to use |
|---|---|---|
| `"person"` | `PersonSamplerParams` | **Preferred.** Locale downloaded to `~/.data-designer/managed-assets/datasets/` by default. |
| `"person_from_faker"` | `PersonFromFakerSamplerParams` | Fallback when locale not downloaded. Basic names/addresses via Faker, not demographically accurate. |

## Available persona datasets

Before using, always check for installed persona datasets with this command:

```bash
data-designer agent state persona-datasets
```

## Usage

The sampled person column is a nested dict. You can keep it as-is in the final dataset, or set `drop=True` to remove it and extract only the fields you need via `ExpressionColumnConfig`:

```python
# Keep the full person dict in the output
config_builder.add_column(dd.SamplerColumnConfig(
    name="person", sampler_type="person",
    params=dd.PersonSamplerParams(locale="en_US"),
))

# Or drop it and extract specific fields
config_builder.add_column(dd.SamplerColumnConfig(
    name="person", sampler_type="person",
    params=dd.PersonSamplerParams(locale="en_US"), drop=True,
))
config_builder.add_column(dd.ExpressionColumnConfig(
    name="full_name",
    expr="{{ person.first_name }} {{ person.last_name }}", dtype="str",
))
```

## PersonSamplerParams

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `locale` | `str` | `"en_US"` | Must be a downloaded managed-dataset locale |
| `sex` | `"Male" \| "Female" \| None` | `None` | Filter by sex |
| `city` | `str \| list[str] \| None` | `None` | Filter by city |
| `age_range` | `list[int]` | `[18, 114]` | `[min, max]` inclusive |
| `select_field_values` | `dict[str, list[str]] \| None` | `None` | Flexible field filtering |
| `with_synthetic_personas` | `bool` | `False` | Append Big Five + persona fields |

## PersonFromFakerSamplerParams

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `locale` | `str` | `"en_US"` | Any Faker-supported locale |
| `sex` | `"Male" \| "Female" \| None` | `None` | Filter by sex |
| `city` | `str \| list[str] \| None` | `None` | Filter by city |
| `age_range` | `list[int]` | `[18, 114]` | `[min, max]` inclusive |

## Person fields (keys in sampled dict)

**Standard fields:** `uuid`, `first_name`, `middle_name`, `last_name`, `sex`, `age`, `birth_date`, `marital_status`, `postcode`, `city`, `region`, `country`, `locale`, `education_level`, `bachelors_field`, `occupation`, `national_id`, `street_name`, `street_number`, `email_address`, `phone_number`

**Locale-specific:** `unit`/`state` (US), `area`/`prefecture`/`zone` (JP), `race` (BR), `district`/`education_degree`/`first_language`/`second_language`/`third_language` (IN), `religion` (BR, IN)

**Persona fields** (when `with_synthetic_personas=True`): `persona`, `detailed_persona`, `cultural_background`, `career_goals_and_ambitions`, `hobbies_and_interests`, `skills_and_expertise`, Big Five scores (`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`), plus domain personas (`professional_persona`, `finance_persona`, `healthcare_persona`, etc.)
